import os
import sys
from ruamel.yaml import YAML

from utils.merge_lora_model import merge_lora_to_base_model
from utils.logger import logger
from utils.search_checkpoint import list_training_outputs,search_latest_checkpoint,select_training_model


#读取配置文件config.yaml
yaml = YAML()
with open('config.yaml', 'r') as f:
    config = yaml.load(f)           # 导入配置文件config.yaml

# mode_path = "/home/liangshuqiao/models/DeepSeek-R1-Distill-Qwen-7B"

model_path = config['file_load']['model_path']
# 交互式选择模型
base_output_dir = config['file_load']['lora_output_path']
selected_model_path,selected_folder = select_training_model(base_output_dir)


lora_path = search_latest_checkpoint(selected_model_path)

merge_model_path = config['file_load']['merge_model_path']

if lora_path is not None:
    try:
        merge_lora_to_base_model(model_path,lora_path,merge_model_path)
    except Exception as e:
        logger.error(f"merge error:{e}")
        raise
