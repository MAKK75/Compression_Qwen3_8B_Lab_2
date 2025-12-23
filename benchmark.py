import torch
import os
import gc
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from awq import AutoAWQForCausalLM
from datasets import load_dataset
from huggingface_hub import model_info

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

seed_everything(42)

ORIG_MODEL_ID = "Qwen/Qwen3-8B"
COMP_MODEL_ID = "NOVORDSEC/qwen3-8b-awq-int4" 
FRACTION = 0.2  

def get_model_size_gb(model_id):
    try:
        info = model_info(model_id, files_metadata=True)
        size = sum(s.size for s in info.siblings if s.size and any(ext in s.rfilename for ext in ['.safetensors', '.bin']))
        return size / (1024**3)
    except Exception as e:
        print(f"Ошибка получения размера для {model_id}: {e}")
        return 0.0

def run_mmlu_benchmark(model, tokenizer, fraction=0.2, desc="Benchmarking"):
    subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 
        'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 
        'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 
        'professional_law', 'professional_medicine', 'professional_psychology', 
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]

    try:
        device = model.device
    except AttributeError:
        device = next(model.parameters()).device

    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]
    
    detailed_results = {}
    total_correct = 0
    total_questions = 0
    
    model.eval()
    for subject in tqdm(subjects, desc=desc):
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            num_samples = max(1, int(len(dataset) * fraction))
            dataset = dataset.select(range(num_samples))
            
            sub_correct = 0
            for item in dataset:
                prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
                prompt += f"{item['question']}\n"
                prompt += f"(A) {item['choices'][0]}\n(B) {item['choices'][1]}\n(C) {item['choices'][2]}\n(D) {item['choices'][3]}\n"
                prompt += "Answer:"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.inference_mode():
                    logits = model(**inputs).logits[0, -1, :]
                    relevant_logits = logits[choice_tokens]
                    pred = torch.argmax(relevant_logits).item()
                    
                    if pred == item['answer']:
                        sub_correct += 1
            
            acc = sub_correct / num_samples
            detailed_results[subject] = acc
            total_correct += sub_correct
            total_questions += num_samples
        except Exception as e:
            print(f"Ошибка в теме {subject}: {e}")
            
    return (total_correct / total_questions if total_questions > 0 else 0), detailed_results


size_orig = get_model_size_gb(ORIG_MODEL_ID)
size_comp = get_model_size_gb(COMP_MODEL_ID)
ratio = size_orig / size_comp if size_comp > 0 else 0
print(f"Original: {size_orig:.2f} GB | Compressed: {size_comp:.2f} GB | Ratio: {ratio:.2f}x")

print("\n>>> Оценка ОРИГИНАЛЬНОЙ модели...")
tokenizer = AutoTokenizer.from_pretrained(ORIG_MODEL_ID, trust_remote_code=True)
model_orig = AutoModelForCausalLM.from_pretrained(
    ORIG_MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

orig_acc, orig_detailed = run_mmlu_benchmark(model_orig, tokenizer, fraction=FRACTION, desc="Original Model")

del model_orig
gc.collect()
torch.cuda.empty_cache()

print("\n>>> Оценка СЖАТОЙ модели (AWQ)...")
model_quant = AutoAWQForCausalLM.from_quantized(
    COMP_MODEL_ID, 
    fuse_layers=True, 
    device_map="auto", 
    trust_remote_code=True
)
tokenizer_quant = AutoTokenizer.from_pretrained(COMP_MODEL_ID, trust_remote_code=True)

comp_acc, comp_detailed = run_mmlu_benchmark(model_quant, tokenizer_quant, fraction=FRACTION, desc="Compressed Model")

drop = (orig_acc - comp_acc) / orig_acc if orig_acc > 0 else 0
score = ratio / (1 + max(0, drop))

print("\n" + "="*40)
print(f"ИТОГИ ЭТАПА 1:")
print(f"Compression Ratio: {ratio:.4f}")
print(f"Baseline Accuracy: {orig_acc:.4f}")
print(f"Compressed Accuracy: {comp_acc:.4f}")
print(f"Performance Drop: {drop*100:.2f}%")
print(f"ФИНАЛЬНЫЙ SCORE: {score:.4f}")
print("="*40)

subjects = sorted(list(orig_detailed.keys()))
comparison_data = []
for s in subjects:
    comparison_data.append({
        "Subject": s,
        "Original_Acc": orig_detailed.get(s, 0.0),
        "Compressed_Acc": comp_detailed.get(s, 0.0),
        "Diff": orig_detailed.get(s, 0.0) - comp_detailed.get(s, 0.0)
    })

df_final = pd.DataFrame(comparison_data)
df_final.to_csv("mmlu_comparison_detailed.csv", index=False)
print(f"Детальный отчет сохранен в mmlu_comparison_detailed.csv")