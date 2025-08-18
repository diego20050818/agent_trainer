from unsloth import FastLanguageModel
import torch
#from prompt import alpaca_prompt
from rich import print as rprint
import pandas as pd
import time
import json
import re
from ruamel.yaml import YAML

yaml = YAML()
with open('config.yaml', 'r') as f:
    config = yaml.load(f)

max_seq_length = 256
dtype = None
load_in_4bit = True
model_save_path = config['file_load']['save_model_path']
#abstract_model_save_path = 'abstract_nlp_palm2.0_pretrained_chinese-base'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_save_path, 
    max_seq_length = max_seq_length, 
    dtype = dtype,     
    load_in_4bit = load_in_4bit, 
)

""" identify = "你的名字是“琪亚娜”,你是一个圣芙蕾雅医美的客服。" \
"你从和客户的对话中提取诸如电话号码，意向产品等重要讯息。" \
"你准确的根据知识库中的知识提供专业而易于理解的讯息，接纳客户的一切情绪。" \
"并适时给客户提供情绪价值，不要输出模棱两可的输出！" \
"只需要输出回答，其他的话不要多说！" \
"多用“亲”等字眼拉进与客户的距离"
 """
identify = "请根据问题做出回答，只输出答案。"

FastLanguageModel.for_inference(model) 
conversation_history = []
conversation_abstract = []
EOS_TOKEN = tokenizer.eos_token
while True:
    conversation_history_length = len(conversation_history)
    rprint("对话长度："+str(conversation_history_length))
    user_ID = 'diego'
    user_input = input("user:")
    
    if user_input != 'q':
        try:
            conversation_history.append({"role":"user","content":user_input})
            
            conversation_text = ""
            for message in conversation_history[-4:]:
                conversation_text += f"{message['role']}:{message['content']}\n"
            #inputs = tokenizer(user_input, return_tensors = "pt").to("cuda")
            inputs = tokenizer(
                [config['qwen_prompt_output'].format(
                    identify,
                    None,
                    user_input,
                    ""
                )], 
                return_tensors = "pt").to("cuda")

            from transformers import TextStreamer
            #text_streamer = TextStreamer(tokenizer)
            #_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = max_seq_length)
            _ = model.generate(**inputs, max_new_tokens = max_seq_length)
            # 获取模型生成的响应
            response = tokenizer.decode(_[0], skip_special_tokens=True)

            # 提取response标签
            match = re.search(r'<response>(.*?)</response>',response,re.DOTALL)
            if match:
                response_contents = match.group(1).strip()
            else:
                response_contents = response.split("system:")[-1].strip()
            # 将模型生成的响应添加到对话历史
            conversation_history.append({"role": "assistant","content": response_contents})
        except Exception as e:
                print(e)
                pass
        print(response_contents)
        pass
        

    else:
        break
t = time.localtime()
log_conversation = {'user_ID':user_ID,'version':time.asctime(),"model":model_save_path,"conversation":conversation_history}

import os

if os.path.exists('conversation_history.json') and os.path.getsize('conversation_history.json') > 0:
    with open('conversation_history.json', 'r') as f:
        data = json.load(f)
        if not isinstance(data, list):
            data = []
else:
    data = []
    data.append(log_conversation)
with open('conversation_history.json', 'w') as f:
    json.dump(data,f,ensure_ascii=False,indent=4)

