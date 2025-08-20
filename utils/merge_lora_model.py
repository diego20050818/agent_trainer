import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.logger import logger

def merge_lora_to_base_model(base_model_path:str,output_lora_dir:str,merged_model_path:str):
    '''
    用来合并模型和lora参数
    
    Args:
        base_model_path:基础模型的路径
        output_lora_path:训练完成后lora的checkpoint路径（需要指定到具体的checkpoint)
        merge_model_path:合并之后的模型路径
    
    Returns:
        None
    '''
    
    # 加载基础模型
    logger.info("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    logger.info("正在加载LoRA权重...")
    lora_model = PeftModel.from_pretrained(
        base_model, 
        output_lora_dir
    )
    
    # 合并模型
    logger.info("正在合并模型...")
    merged_model = lora_model.merge_and_unload()
    
    # 加载tokenizer
    logger.info("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 创建保存目录
    os.makedirs(merged_model_path, exist_ok=True)
    
    # 保存合并后的模型
    logger.info("正在保存合并后的模型...")
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    
    logger.info(f"模型合并完成，已保存至: {merged_model_path}")
