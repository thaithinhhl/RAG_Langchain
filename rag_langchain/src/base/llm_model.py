import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os
import gc

# Kiểm tra xem accelerate đã được cài đặt chưa
has_accelerate = False
try:
    import importlib
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    if has_accelerate:
        print("Đã tìm thấy thư viện accelerate!")
except:
    pass

try:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
except Exception as e:
    nf4_config = None

def get_hf_llm(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
               max_new_tokens = 512,
               use_auth_token = None,
               **kwargs):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    if use_auth_token is None and "llama" in model_name.lower():
        use_auth_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    
    if use_auth_token:
        model_kwargs["token"] = use_auth_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=use_auth_token
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )
    
    return llm

