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

history = []
# print(history)

choose = input("1-普通对话  2-预设对话:")

if choose == '1':
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
            return_dict=True,
            enable_thinking=False   #开启思维链
        ).to("cuda")

        gen_kwargs = {
            "max_length": 4096,
            # "do_sample": True,
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

if choose == '2':
    history = []

    for i, pair in enumerate(conversation_pair):
        system_prompt = config['system_prompt']

        # 如果是用户轮次，触发模型生成
        if pair["role"] == "user":
            user_input = pair["content"]
            history.append({"role": "user", "content": user_input})

            messages = [{"role": "system", "content": system_prompt}] + history

            inputs = tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False   #开启思维链
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
                print(f"轮次 {i//2 + 1}")
                print("用户输入:")
                print(user_input)

                # 模型生成
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                output_ids = model.generate(
                    **inputs,
                    **gen_kwargs,
                    streamer=streamer
                )

                generated_text = tokenizer.decode(
                    output_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                history.append({"role": "assistant", "content": generated_text})

                # 找到对应的期望 assistant 回复
                expected = None
                if i + 1 < len(conversation_pair) and conversation_pair[i+1]["role"] == "assistant":
                    expected = conversation_pair[i+1]["content"]

                print("\n模型输出:")
                print(generated_text)
                if expected:
                    print("\n预计回答:")
                    print(expected)
                print("="*50 + "\n")
