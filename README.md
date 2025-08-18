# agent-fine-tuning-trainer

模型微调训练脚本，一坨，其他不赘述了

## 环境准备
- 建议使用uv进行包管理，因为主播就是用的uv，当然没有也没关系
- python版本建议3.10以上，3.12以下（许多包对3.12的兼容性不好）
- 平台：windows/Linux，建议Linux

### 使用UV进行包管理的方式
安装uv包管理器
```bash
pip install uv

cd ./agent-fine-tuning-trainer
```

1. 进入项目,创建环境（名为env)

```bash
uv venv env --python 3.10
```

2. 从`pyproject.toml`中安装依赖

```bash
uv sync
```

3. 激活虚拟环境

```bash
# windows
env/Scripts/activate

# Linux
source env/bin/Scripts
```

**如果`pyproject.toml`的方法失效，那么可以使用`requirements.txt`进行安装**，命令如下：

```bash
uv pip install -r requirements.txt
```

### 使用conda 进行包管理的方式（我还没试过，理论上可以的）

```bash
# 创建虚拟环境
conda create env -n trainer python==3.10.0

#激活环境
conda activate trainer

# 安装依赖
pip install -r requirements.txt
```

## 开始训练
*目前这个脚本还并没有很完善，所以可能还是有点怪*

**注意！相关的超参都可以在根目录下的config.yaml中配置，不要硬编码在代码中**

### 文件
- train开头的文件是微调用的，notebook文件可以不管，只是代码框架参考
- test开头的文件是训练之后测试用的，会自动寻找model_output（也就是微调之后的lora保存的文件夹）中最大的那个checkpoint文件进行测试
- main.py现在还没什么用，以后可以用来变成智能识别模型信息并调用合适脚本自动训练和测试的统一入口
- file.log中保存了所有训练、测试等关键步骤的关键信息，包括训练结果
- utils文件夹下是工具包，里面有个logger文件，如果实在看不惯我独特的日志输出，可以将每个训练、测试脚本的`from utils.logger import logger`换成`from loguru import logger`但是file.log不会再记录你的日志信息，如果仅仅是不想要流式输出的日志，可以将utils/logger.py中 以下代码的delay调成0
```python
#87行
typing_handler = TypingConsoleHandler(sys.stdout, delay=0.03)  # 控制每个字母的延迟
```


### 目前训练过的模型

- DeepSeek-R1-Distill-Qwen-7B
- qwen2-14b
- qwen2.5

qwen3因为手上的v100不支持FB8的量化所以还没有试

### 数据集格式

根目录下的`train_deepseek.py`以及`train_qwen.py`都使用以下数据集格式

```json
[
    {
        "conversation_id": 1,
        "system_prompt": "你是一个医美客服助手",
        "conversation_pair": [
        {
            "role": "user",
            "content": "你好，做一次超声炮多少钱？"
        },
        {
            "role": "assistant",
            "thinking": "确认顾客想改善的问题",
            "content": "您好，超声炮现在活动XXXX全面部，您是对面部那个部分不满意呢？"
        },
        ]
    },
]
```

