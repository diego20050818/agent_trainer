
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
import torch
from peft import PeftModel

from utils.logger import logger
from utils.search_checkpoint import select_training_model,search_latest_checkpoint
# from loguru import logger

import sys
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 
logger.info(f"========{__name__}  {current_time}========")
logger.info(f"当前环境：{sys.executable}")
logger.info("开始进行模型测试")

from datasets import Dataset,load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

import datetime 
from ruamel.yaml import YAML
import torch
import json

#读取配置文件config.yaml
yaml = YAML()
with open('config.yaml', 'r') as f:
    config = yaml.load(f)           # 导入配置文件config.yaml

# mode_path = "/home/liangshuqiao/models/DeepSeek-R1-Distill-Qwen-7B"

mode_path = config['file_load']['model_path']
# 交互式选择模型
base_output_dir = config['file_load']['lora_output_path']
selected_model_path,selected_folder = select_training_model(base_output_dir)


lora_path = search_latest_checkpoint(selected_model_path)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)


conversation_pair = [
    {"role": "user", "content": "脱一下毛"},
    {"role": "assistant", "thinking": "询问改善部位", "content": "您好～了解哪个部位的脱毛呢"},
    {"role": "user", "content": "大腿和小腿"},
    {"role": "assistant", "thinking": "询问改善时间", "content": "可以的，您是想什么时候脱呢 我给您预约一下"},
    {"role": "user", "content": "大概要做几次，是不是激光的吗？"},
    {"role": "assistant", "thinking": "给出解答，探索顾客意向时间", "content": "冰点脱毛呢，要看您的毛发旺盛程度 有的人 两三次 有的五六次，\n一般脱干净了 就不会再长了，您是想什么时候脱呢"},
    {"role": "user", "content": "我这种腿毛比较旺盛的呢？能脱得干净吗？"},
    {"role": "assistant", "thinking": "解答问题，消除顾客疑虑", "content": "都能脱干净 就是次数问题，一次比一次更稀疏"},
    {"role": "user", "content": "明天我能去吗"},
    {"role": "assistant", "thinking": "套电", "content": "可以呀 方便加您个联系方式吗"},
    {"role": "user", "content": "13156654334"},
    {"role": "assistant", "thinking": "", "content": "好的呢"}
]


def test_from_pre(conversation_pair:list):
    """从预设的问答对来测试模型表现
    
    Args:
        conversation_pair (list): 问答对

    Examples:
        >>> {"role": "assistant", "thinking": "", "content": "好的呢"}
        >>> {"role": "user", "content": "13156654334"},

    """
    history = []  #  用列表保存完整对话

    for i in range(1, len(conversation_pair), 2):  # 每轮assistant
        user_input = conversation_pair[i-1]["content"]
        real_think = conversation_pair[i].get("thinking", "")
        real_content = conversation_pair[i]["content"]
        system_prompt = config['system_prompt']

        # 把历史对话 + 当前输入一起构造
        history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": system_prompt}] + history

        inputs = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        gen_kwargs = {
            "max_length": 2048,
            "do_sample": True,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 0.2
        }

        with torch.no_grad():
            print("="*50)
            print("用户输入:")
            print(user_input)
            print("模型输出:<think>", end='', flush=True)

            # 使用 TextStreamer 实现流式打印
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            output_ids = model.generate(
                **inputs,
                **gen_kwargs,
                streamer=streamer
            )

            # 解析生成的内容（只截取生成部分）
            generated_text = tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # 把模型回答加进 history，供后续轮次使用
            history.append({"role": "assistant", "content": generated_text})

            print("\n\n样本真实思维链 + 输出:")
            print(f"<think>{real_think}</think> \n{real_content}")
            print("="*50)
            print("\n")


# 在函数外部定义 history 变量，使其在多次调用之间保持状态
history = []  # 用列表保存完整对话

def test_from_user(user_input:str):
    """用户输入以获取模型输出，主观评价

    Args:
        user_input (str): 用户输入字符
    """
    global history
    # print(history)

    system_prompt = config['system_prompt']

    # 把历史对话 + 当前输入一起构造
    history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": system_prompt}] + history

    inputs = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    gen_kwargs = {
        "max_length": 4096,
        "do_sample": True,
        "top_k": 1,
        "top_p": 0.9,
        "temperature": 0.2
    }

    with torch.no_grad():
        print("="*50)
        print("用户输入:")
        print(user_input)
        print("模型输出:", end='', flush=True)

        # 使用 TextStreamer 实现流式打印
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
            streamer=streamer
        )

        # 解析生成的内容（只截取生成部分）
        generated_text = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 把模型回答加进 history，供后续轮次使用
        history.append({"role": "assistant", "content": generated_text})

        print("="*50)
        print("\n")



if __name__ == '__main__':
    test_from_pre(conversation_pair=conversation_pair)

    # while True:

    #     user_input = input("用户：")

    #     import sys
    #     if user_input.strip().lower() == 'q':
    #         print("对话将终止，明天见！")
    #         break

    #     test_from_user(user_input)