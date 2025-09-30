import subprocess
print("Hello from agent-fine-tuning-trainer!")
print("请选择操作:")

test_origin = 'test_origin_model.py'
test_qwen_lora = 'test_QwenForCausalLM.py'
train_qwen_lora = 'train_QwenForCausalLM.py'
merge_model = 'merge_model.py'
check_model = "check_model.py"

op_dict = {
    '1':[test_origin,'测试原始模型'],
    '2':[test_qwen_lora,'测试微调效果'],
    '3':[train_qwen_lora,'训练模型'],
    '4':[merge_model,'合并模型'],
    '5':[check_model,'查看模型']
}


def main(operate:str):
    script = op_dict[operate][0]
    subprocess.run(['python',script])


if __name__ == "__main__":
    while True:

        for key,value in op_dict.items():
            print(f"\t{key}'\t\t{value[1]}")
        print(f"\tq\t\t退出")

        operate = input("操作:")
        
        if operate.strip().lower() == 'q':
            break
        main(operate)
