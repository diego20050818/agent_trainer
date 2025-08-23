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
```bash
|-- file.log        #你的日志文件，需要from utils.logger import logger 自定义日志启用
|-- main.py         #目前没什么大用，准备拿来当做选择模型、训练、评估、合成完整模型的封装脚本
|-- pyproject.toml  # 适用uv安装的配置文件，没有这个东西你别想 uv sync成功   
|-- requirements.txt    # 传统的环境依赖列表
|-- test_QwenForCausalLM.py #qwen for causallm框架的测试脚本（qwen2,qwen2.5,qwen3-8b适用）
|-- test_origin_model.py    #测试原始、没有lora的模型
|-- train_QwenForCausalLM.py    #qwen for causallm框架的训练脚本（qwen2,qwen2.5,qwen3-8b适用）
|-- train_gpt.py    # gpt-oss的训练脚本，也是目前最简洁的框架
|-- check_model.py # 用来检查模型信息
|-- train_arg   #用来存放每个模型的训练参数，要训什么模型就传入什么模型的配置脚本，互相不影响
|   |-- deepseek-r1-qwen-7b.yaml
|   |-- gemma-3.yaml
|   |-- gpt-oss.yaml
|   |-- qwen2_5-14b.yaml
|   `-- qwen3-8b.yaml
|-- trainlog    #适用Tensorboard的训练日志（实时）
|   |-- events.out.tfevents.1755432485.gpu04.2229690.0
|-- utils   #自定义工具包
|   |-- __init__.py
|   |-- load_dataset.py # 导入数据集、为不同的模型做数据集map函数，详细信息见class doc
|   |-- logger.py   #自定义日志，翁法罗斯风格的哦
|   |-- merge_lora_model.py     #将训练完成并通过检查的模型和lora训练成果合并为完整的模型
|   |-- search_checkpoint.py    #可以自动寻找lora导出目录下的模型列表和最新的检查点文件
|   |-- tools.py     #一些比较杂的重复性高的方法存在这里
|   `-- visualize_train.py  # 没什么大用，本来想用来可视化训练数据的，写炸钢了准备删掉
`-- uv.lock
```

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
- *还有好几个但是懒得写了谢谢*


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
可以根据需要增添内容（构想中会多一个键值对:`key=场景,value=新人留联`之类的，如果仔细看源代码（utils/load_dataset.py）部分的话，会发现其实所有的场景相关全部被硬编码了

### 配置文件
配置文件分为两种：根目录下的`config.yaml`和谓语`./train_arg`文件夹下的`'模型名称'.yaml`
其中config.yaml定义了基本的模型路径，数据集路径，数据集分割方式，需要修改模型的时候，需要同时把以下键值对修改：
1. `model_name`与导出checkpoint的目录名称相关，不会直接影响训练，但是尽量统一名称以便查找
2. `model_path`模型文件的存放路径，不需要解释，但是请在训练之前**检查模型文件完整性**（可以使用check_model.py来检查模型完整性，如果报错了，那么模型不受支持、硬件不受支持或者模型文件不完整）
3. `save_model_path`合并模型之后的文件夹，如果不设定的话可能会把以前导出的模型覆盖掉

其他配置可以参考注释，注意**数据集分割方法不需要修改**

### 训练脚本
训练脚本用的是huggingface的框架，具体的写法可以参考`train_gpt.py`,这个脚本写的是最清楚的，只需要在配置文件中修改键值，就可以一键训练

<h1 align="left" style="color:red ; font-size:50px">注意注意</h1>

```python
'''
train_gpt.py
'''

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
    fuc=data.process_gpt,       # ->需要看清楚是什么模型需要的映射函数，见下
    remove_columns=['conversation_id', 'system_prompt', 'conversation_pair']
)
logger.info("数据集加载完成")

'''
utils/dataset_loader.py
'''
# ...extra code
class data_loder:
    ...
    def process_qwen(self,example):     #->给千问（有思维链的）数据集映射函数
        """
        构造多轮对话数据的实例
        原理：直接构造包括多轮对话中所有机器人回复内容的标签，
        既充分地利用了所有机器人的回复信息，同时也不存在拆重复计算。

        inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
        labels = <-100> <assistant1> <-100> <assistant2> <-100> <assistant3>
        """
        pass

    def process_gpt(self,example):      # ->给gpt-oss的数据集映射函数
        """针对gpt-oss的多轮对话映射函数，注意这个函数并不会处理think块

        Args:
            example (_type_): 传入dataset.map函数默认
        """
        pass
    # ...extra code
```

也就是说，遇到不同的模型，映射函数在`utils/data_loader.py`中进行扩展，以便复用，而不是在训练脚本中编写，反过来，在训练脚本中我们直接从data_loader类中提取映射方法。
后续也可以尝试使用`getattr(data_loader,config['map_fuc'])`这一类方式也把映射函数放进yaml配置文件中显式配置
- 参数配置好之后直接`python train_gpt.py`即可

### 评估
每个评估脚本都是差不多的，都会从配置文件中读取你的模型，并且遍历你的lora输出文件夹，并让你选择需要评估什么模型。
**也就是说，如果你的基础模型是qwen3,但是你选择了gpt的lora文件夹，就有可能会报错，而且这里我没有写检查**
#### 关于`test_origin_model`和`test_{model_struct_name}`的区别
一个是直接测试基础模型，没有微调的（用来和微调之后的进行对比），一个是读取原模型和lora的检查点文件进行微调之后的测试的，测试分为两种，一种的预定的多轮对话输出，一种是主观的用户打字问答输出，两个都是流式输出
开关方式如下：
```python
'''
test_QwenForCausalLM.py
'''
if __name__ == '__main__':
    test_from_pre(conversation_pair=conversation_pair)#需要用预定的数据集微调就取消注释这个

    # while True:           需要自己试一下的就把这个while True块给释放掉

    #     user_input = input("用户：")

    #     import sys
    #     if user_input.strip().lower() == 'q':
    #         print("对话将终止，明天见！")
    #         break

    #     test_from_user(user_input)

```

