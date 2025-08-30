# gpt-oss的训练脚本
import torch
from ruamel.yaml import YAML
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig,TaskType,get_peft_model
from datasets import Dataset

from utils.logger import logger,TrainingLogCallback
from utils.load_dataset import dataset_loder
from utils.tools import train_arg_printer,generate_outputdir

logger.info("==========模型微调脚本启动==========")
logger.info("模块导入完成")

# ====== 导入配置 =======
config_path = "config.yaml"
train_path = "train_arg/gpt-oss.yaml"

yaml = YAML()
with open(config_path,'r') as f:
    config = yaml.load(f)

with open(train_path,'r') as f:
    train_arg = yaml.load(f)

logger.info("配置导入完成")


# ======= 导入模型和分词器 ======
model = AutoModelForCausalLM.from_pretrained(
    config['file_load']['model_path'],
    trust_remote_code=False,
    torch_dtype=getattr(torch,train_arg['dtype'].split('.')[-1]),
    device_map=None,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(
    config['file_load']['model_path'],
    trust_remote_code=False
)
logger.info("模型和分词器加载完成")
# ======= 导入数据集 ========
special_token = {
    'bos':tokenizer.bos_token,
    'eos':tokenizer.eos_token
}

data = dataset_loder(
    dataset_path=config['file_load']['dataset_path'],
    dataset_class='json',
    max_length=train_arg['max_seq_length'],
    tokenizer=tokenizer,
    special_token=special_token
)

dataset = data.dataset_map(
    fuc=data.process_gpt,
    remove_columns=['conversation_id', 'system_prompt', 'conversation_pair']
)
logger.info("数据集加载完成")

#======训练参数和训练部分======

# 从配置文件获取LoRA参数
lora_config = LoraConfig(
    task_type=getattr(TaskType, train_arg['task_type']),
    target_modules=train_arg['target_modules'],
    inference_mode=train_arg['interface_mode'],  # 训练模式
    r=train_arg['r'],  # Lora 秩
    lora_alpha=train_arg['lora_alpha'],  # Lora alpha
    lora_dropout=train_arg['lora_dropout']  # Dropout 比例
)

train_arg_printer(train_arg)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

output_dir = generate_outputdir(config['file_load']['model_name'])
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
    # gradient_checkpointing=train_arg['use_gradient_checkpointing'],
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
    trainer.train()
    logger.info("训练成功！")
    logger.info(f"模型存放位置：{output_dir}")
except Exception as e:
    logger.error(f"训练失败：{e}")
    raise