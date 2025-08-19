from utils.logger import logger
logger.info("开始进行原始模型对话测试")

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from ruamel.yaml import YAML
import torch
import json
logger.info("导入包完成")

# 读取配置文件config.yaml
yaml = YAML()
with open('config.yaml', 'r') as f:
    config = yaml.load(f)
logger.info("配置文件读取完成")

model_name = config['file_load']['model_path']

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

history = []
# print(history)
while True:
    system_prompt = config['system_prompt']
    user_input = input("user:")

    if user_input.strip().lower() == 'q':
        break

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