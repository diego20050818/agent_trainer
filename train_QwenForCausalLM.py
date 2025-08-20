"""适用于qwen2cuasual llm框架的训练脚本

Returns:
    _type_: 模型lora训练文件
"""
import sys
import datetime

from datasets import Dataset, load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from ruamel.yaml import YAML
from rich import print as rprint
import torch

from utils.load_dataset import dataset_loder
from utils.logger import logger,TrainingLogCallback

logger.info("导入包完成")

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 
logger.info(f"========train Qwen2ForCausalLM  {current_time}========")
logger.info(f"当前环境：{sys.executable}")
logger.info("开始进行训练")

config_path = 'config.yaml'
train_arg = 'train_arg/qwen.yaml'
# 读取配置文件config.yaml
yaml = YAML()
with open('config.yaml', 'r') as f:
    config = yaml.load(f)
logger.info("基础配置文件读取完成")

with open(train_arg,'r') as f:
    train_arg = yaml.load(f)
logger.info("训练配置读取完成")

# 从配置文件获取路径
dataset_path = config['file_load']['dataset_path']
model_path = config['file_load']['model_path']
logger.info(f"数据集路径：{dataset_path}")
logger.info(f"模型路径:{model_path}")

# ===== 导入模型 =====
try:
    # 开始读取分词表
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=False
    )
    tokenizer.padding_side = 'right'  # 根据模型特性从右边开始填充
    logger.info("tokenizer读取完成")
except Exception as e:
    logger.error(f"分词表导入失败：{e}")
    raise

try:
    # 开始读取模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=False,
        torch_dtype=getattr(torch, train_arg['dtype'].split('.')[-1]),
        device_map=None,
        low_cpu_mem_usage=True
    )
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.enable_input_require_grads()
    logger.info(f"model dtype:{model.dtype}")
    logger.info("模型导入完成")
except Exception as e:
    logger.error(f"模型导入失败：{e}")
    raise

# ===== 导入和处理数据集 =====

# 定义开头和结尾的特殊token
special_tokens = {
    "bos": tokenizer.bos_token,
    "eos": tokenizer.eos_token
}

dataset_loder = dataset_loder(
    dataset_path=dataset_path,
    dataset_class='json',
    max_length=train_arg['max_seq_length'],
    tokenizer=tokenizer,
    **special_tokens
)
dataset = dataset_loder.dataset_map(
    dataset_loder.process_qwen,
    remove_columns=[
            'conversation_id', 'system_prompt', 'conversation_pair',
        ]
)



"""
attention_mask 用来控制 哪些 token 是有效的输入（padding 才会 0）
labels 控制 哪些 token 参与 loss 计算（-100 不算 loss）
"""


# ===== 训练参数和训练部分 =====
from peft import LoraConfig, TaskType, get_peft_model

# 从配置文件获取LoRA参数
lora_config = LoraConfig(
    task_type=getattr(TaskType, train_arg['task_type']),
    target_modules=train_arg['target_modules'],
    inference_mode=train_arg['interface_mode'],  # 训练模式
    r=train_arg['r'],  # Lora 秩
    lora_alpha=train_arg['lora_alpha'],  # Lora alpha
    lora_dropout=train_arg['lora_dropout']  # Dropout 比例
)

print(lora_config)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 生成带时间戳的输出目录
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
model_name = config['file_load']['model_name']
output_dir = f"./output/{model_name}{current_time}"

# 从配置文件获取训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=train_arg['batch_size'],
    gradient_accumulation_steps=train_arg['gradient_accumulator_steps'],
    logging_steps=1,
    num_train_epochs=train_arg['epoch'],
    save_steps=train_arg.get('save_steps', 100),
    learning_rate=float(train_arg['learning_rate']),
    save_on_each_node=True,
    gradient_checkpointing=train_arg['use_gradient_checkpointing'] != "none",
    # 添加这些参数以避免meta tensor问题
    remove_unused_columns=True,
    dataloader_pin_memory=False,
    lr_scheduler_type=train_arg['lr_scheduler_type'],
    warmup_steps=train_arg['warmup_steps'],
     # 添加日志相关参数
    logging_dir=config['file_load']['logging_path'],  # 日志目录
    logging_strategy="steps",  # 按步骤记录日志
    logging_first_step=True,  # 记录第一步
    report_to=["tensorboard"],  # 报告到tensorboard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[TrainingLogCallback()]
)

logger.info("开始训练！")
try:
    logger.info(f"批次大小  : {train_arg['batch_size']}")
    logger.info(f"训练轮数  : {train_arg['epoch']}")
    logger.info(f"学习率    : {train_arg['learning_rate']}")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"模型路径  : {model_path}")
    trainer.train()
    logger.info("训练成功！")
    logger.info(f"模型存放位置：{output_dir}")
except Exception as e:
    logger.error(f"训练失败：{e}")
    raise