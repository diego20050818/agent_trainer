from utils.logger import logger,TrainingLogCallback
import sys
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 
logger.info(f"========{__name__}  {current_time}========")
logger.info(f"当前环境：{sys.executable}")
logger.info("开始进行训练")

from datasets import Dataset, load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

from ruamel.yaml import YAML
from rich import print as rprint
import torch
import json
logger.info("导入包完成")

# 读取配置文件config.yaml
yaml = YAML()
with open('config.yaml', 'r') as f:
    config = yaml.load(f)
logger.info("配置文件读取完成")

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
        torch_dtype=getattr(torch, config['training_arg']['dtype'].split('.')[-1]),
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
try:
    # 从文件路径读取数据集
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset['train']
    logger.info("读取数据集成功")
except Exception as e:
    logger.error(f"读取数据集失败：{e}")
    raise

# 定义开头和结尾的特殊token
BOS = tokenizer.bos_token
EOS = tokenizer.eos_token

con_have_no_pair = 0

def process(example):
    """
    构造多轮对话数据的实例
    原理：直接构造包括多轮对话中所有机器人回复内容的标签，
    既充分地利用了所有机器人的回复信息，同时也不存在拆重复计算。

    inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
    labels = <-100> <assistant1> <-100> <assistant2> <-100> <assistant3>
    """
    global con_have_no_pair  # 记录异常对话的计数器
    max_length = config['training_arg']['max_seq_length']

    system_prompt = (
        example['system_prompt'] 
        if example['system_prompt'] is not None 
        else "你是一个医美客服助手，请根据用户问题做出回答"
    )
    system_prompt = system_prompt.format("新用户套电")
    conversation_pair = example['conversation_pair']

    input_ids, labels = [], []

    # 添加system prompt
    sys_text = f"{BOS}system:{system_prompt}{EOS}"
    sys_tokens = tokenizer.encode(sys_text, add_special_tokens=False)
    input_ids.extend(sys_tokens)
    labels.extend([-100] * len(sys_tokens))

    if conversation_pair is not None:
        # 遍历对话
        for turn in conversation_pair:
            if turn["role"] == "user":
                text = f"{BOS}用户：{turn['content']}{EOS}"
                tokens = tokenizer.encode(text, add_special_tokens=False)
                input_ids.extend(tokens)
                labels.extend([-100] * len(tokens))   # 用户部分不学习

            elif turn["role"] == "assistant":
                # 如果想保留思考过程
                if turn.get("thinking", ""):
                    think_text = f"{BOS}助手思考：<think>{turn['thinking']}</think>" #因为下面还有输出，并且已经使用内置的think作为特殊token包裹所以这里没有EOS
                    think_tokens = tokenizer.encode(think_text, add_special_tokens=False)
                    input_ids.extend(think_tokens)
                    labels.extend([-100] * len(think_tokens))  # 思考过程不学习

                # 助手最终回复 -> 要作为 label
                resp_text = f"{BOS}助手回复：{turn['content']}{EOS}"
                resp_tokens = tokenizer.encode(resp_text, add_special_tokens=False)
                input_ids.extend(resp_tokens)
                labels.extend(resp_tokens)  # 学习助手回复
    else:
        con_have_no_pair += 1

    # 截断
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    # attention mask: 1表示有效token
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

"""
attention_mask 用来控制 哪些 token 是有效的输入（padding 才会 0）
labels 控制 哪些 token 参与 loss 计算（-100 不算 loss）
"""

if con_have_no_pair > 0:
    logger.warning(f"一共有{con_have_no_pair}条对话没有信息")

try:
    dataset = dataset.map(
        process,
        remove_columns=[
            'conversation_id', 'system_prompt', 'conversation_pair',
            'conversation id', 'system prompt', 'conversation pair'
        ]
    )
    logger.info("数据处理成功")
except Exception as e:
    logger.error(f"处理数据发生错误：{e}")
    raise

# ===== 训练参数和训练部分 =====
from peft import LoraConfig, TaskType, get_peft_model

# 从配置文件获取LoRA参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=config['training_arg']['target_modules'],
    inference_mode=False,  # 训练模式
    r=config['training_arg']['r'],  # Lora 秩
    lora_alpha=config['training_arg']['lora_alpha'],  # Lora alpha
    lora_dropout=config['training_arg']['lora_dropout']  # Dropout 比例
)

print(lora_config)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 生成带时间戳的输出目录
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
output_dir = f"./output/qwen{current_time}"

# 从配置文件获取训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=config['training_arg']['batch_size'],
    gradient_accumulation_steps=config['training_arg']['gradient_accumulator_steps'],
    logging_steps=1,
    num_train_epochs=config['training_arg']['epoch'],
    save_steps=config['training_arg'].get('save_steps', 100),
    learning_rate=float(config['training_arg']['learning_rate']),
    save_on_each_node=True,
    gradient_checkpointing=config['training_arg']['use_gradient_checkpointing'] != "none",
    # 添加这些参数以避免meta tensor问题
    remove_unused_columns=True,
    dataloader_pin_memory=False,
    lr_scheduler_type=config['training_arg']['lr_scheduler_type'],
    warmup_steps=config['training_arg']['warmup_steps'],
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
    logger.info(f"批次大小  : {config['training_arg']['batch_size']}")
    logger.info(f"训练轮数  : {config['training_arg']['epoch']}")
    logger.info(f"学习率    : {config['training_arg']['learning_rate']}")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"模型路径  : {model_path}")
    trainer.train()
    logger.info("训练成功！")
    logger.info(f"模型存放位置：{output_dir}")
except Exception as e:
    logger.error(f"训练失败：{e}")
    raise