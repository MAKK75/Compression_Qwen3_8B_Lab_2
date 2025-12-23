import torch
import os
import gc
import numpy as np
import random
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import model_info

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_size_gb(model_id):
    try:
        info = model_info(model_id, files_metadata=True)
        siblings = info.siblings
        
        def get_file_size(file_obj):
            if hasattr(file_obj, 'lfs') and file_obj.lfs is not None:
                return file_obj.lfs.size
            if hasattr(file_obj, 'size') and file_obj.size is not None:
                return file_obj.size
            return 0

        safetensors_size = sum(get_file_size(s) for s in siblings if s.rfilename.endswith('.safetensors'))
        bin_size = sum(get_file_size(s) for s in siblings if s.rfilename.endswith('.bin'))
        
        final_size = safetensors_size if safetensors_size > 0 else bin_size
        if final_size == 0: 
            final_size = sum(get_file_size(s) for s in siblings if any(ext in s.rfilename for ext in ['.pt', '.pth', '.ckpt']))
            
        return final_size / (1024**3)
    except Exception as e:
        print(f"Ошибка при получении размера: {e}")
        return 0

def compress_model():
    seed_everything(42)
    model_id = "Qwen/Qwen3-8B" 
    quant_path = "./qwen3-8b-awq"
    
    size_orig = get_model_size_gb(model_id)
    print(f"Оригинальный размер модели: {size_orig:.2f} GB")

    quant_config = { 
        "zero_point": True, 
        "q_group_size": 128, 
        "w_bit": 4, 
        "version": "GEMM" 
    }

    print("\nЗагрузка модели для квантования...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("\nНачало калибровки")
    model.quantize(
        tokenizer, 
        quant_config=quant_config,
        n_parallel_calib_samples=1, 
        max_calib_seq_len=512,       
        max_calib_samples=32
    )

    print(f"\nСохранение модели в {quant_path}")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    size_comp = sum(os.path.getsize(os.path.join(quant_path, f)) for f in os.listdir(quant_path) 
                    if any(ext in f for ext in ['.safetensors', '.bin', '.pt'])) / (1024**3)
    
    ratio = size_orig / size_comp if size_comp > 0 else 0
    
    print("\n" + "="*30)
    print(f"РЕЗУЛЬТАТЫ СЖАТИЯ:")
    print(f"Оригинальный размер: {size_orig:.2f} GB")
    print(f"Сжатый размер: {size_comp:.2f} GB")
    print(f"Compression Ratio: {ratio:.2f}x")
    print("="*30)


if __name__ == "__main__":
    compress_model()