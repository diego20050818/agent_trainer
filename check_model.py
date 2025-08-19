'''
这个脚本是用来检查你现在的模型长什么样的
包括模型的框架
'''
'''
这个脚本是用来检查你现在的模型长什么样的
包括模型的框架
'''
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import logger
import numpy as np

def format_number(num):
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def count_parameters(model, trainable_only=False):
    """计算模型参数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def get_model_info(model_path):
    """获取模型详细信息"""
    logger.info("开始加载模型配置文件...")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return None
    
    try:
        logger.info(f"正在加载模型: {model_path}")
        # 加载模型和tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        logger.info("模型加载完成")
        
        # 获取模型基本信息
        info = {}
        info['model_path'] = model_path
        info['model_type'] = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
        info['model_architecture'] = model.config.architectures if hasattr(model.config, 'architectures') else 'unknown'
        
        # 获取模型参数量
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params
        info['non_trainable_parameters'] = total_params - trainable_params
        info['trainable_ratio'] = trainable_params / total_params if total_params > 0 else 0
        
        # 获取特殊token信息
        special_tokens = {}
        if tokenizer:
            special_tokens['bos_token'] = tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else None
            special_tokens['eos_token'] = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else None
            special_tokens['unk_token'] = tokenizer.unk_token if hasattr(tokenizer, 'unk_token') else None
            special_tokens['pad_token'] = tokenizer.pad_token if hasattr(tokenizer, 'pad_token') else None
            special_tokens['sep_token'] = tokenizer.sep_token if hasattr(tokenizer, 'sep_token') else None
            special_tokens['mask_token'] = tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else None
            special_tokens['vocab_size'] = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'unknown'
        info['special_tokens'] = special_tokens
        
        # 获取模型层数信息
        info['num_hidden_layers'] = getattr(model.config, 'num_hidden_layers', 'unknown')
        info['hidden_size'] = getattr(model.config, 'hidden_size', 'unknown')
        info['num_attention_heads'] = getattr(model.config, 'num_attention_heads', 'unknown')
        
        # 获取量化信息
        info['torch_dtype'] = str(getattr(model.config, 'torch_dtype', 'unknown'))
        info['quantization'] = '4bit' if hasattr(model.config, 'quantization_config') else 'None'
        
        # 获取可训练层的信息
        trainable_layers = []
        total_layers = 0
        for name, param in model.named_parameters():
            total_layers += 1
            if param.requires_grad:
                trainable_layers.append(name)
        
        info['total_layers'] = total_layers
        info['trainable_layers'] = trainable_layers
        info['trainable_layers_count'] = len(trainable_layers)
        
        return info
    
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        return None

def print_model_info(info):
    """打印模型信息"""
    if not info:
        logger.error("无法获取模型信息")
        return
    
    logger.info("=" * 60)
    logger.info("模型详细信息")
    logger.info("=" * 60)
    
    logger.info(f"模型路径: {info['model_path']}")
    logger.info(f"模型类型: {info['model_type']}")
    logger.info(f"模型架构: {info['model_architecture']}")
    
    logger.info("-" * 40)
    logger.info("参数信息:")
    logger.info(f"  总参数量: {format_number(info['total_parameters'])} ({info['total_parameters']:,})")
    logger.info(f"  可训练参数: {format_number(info['trainable_parameters'])} ({info['trainable_parameters']:,})")
    logger.info(f"  冻结参数: {format_number(info['non_trainable_parameters'])} ({info['non_trainable_parameters']:,})")
    logger.info(f"  可训练参数比例: {info['trainable_ratio']:.2%}")
    
    logger.info("-" * 40)
    logger.info("模型结构信息:")
    logger.info(f"  层数: {info['num_hidden_layers']}")
    logger.info(f"  隐藏层大小: {info['hidden_size']}")
    logger.info(f"  注意力头数: {info['num_attention_heads']}")
    logger.info(f"  总层数量: {info['total_layers']}")
    logger.info(f"  可训练层数量: {info['trainable_layers_count']}")
    
    logger.info("-" * 40)
    logger.info("特殊Token:")
    for token_name, token_value in info['special_tokens'].items():
        logger.info(f"  {token_name}: {token_value}")
    
    logger.info("-" * 40)
    logger.info("量化信息:")
    logger.info(f"  数据类型: {info['torch_dtype']}")
    logger.info(f"  量化方式: {info['quantization']}")
    
    logger.info("-" * 40)
    logger.info("可训练层 (前10个):")
    for i, layer_name in enumerate(info['trainable_layers'][:10]):
        logger.info(f"  {i+1}. {layer_name}")
    if len(info['trainable_layers']) > 10:
        logger.info(f"  ... 还有 {len(info['trainable_layers']) - 10} 个可训练层")

def main():
    """主函数"""
    # 读取配置文件
    config_path = "config.yaml"
    logger.info(f"正在读取配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_path = config.get('file_load', {}).get('model_path', '')
        if not model_path:
            logger.error("配置文件中未找到模型路径")
            return
            
        logger.info(f"从配置文件中提取到模型路径: {model_path}")
        
        # 获取模型信息
        model_info = get_model_info(model_path)
        
        # 打印模型信息
        print_model_info(model_info)
        
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"配置文件格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()