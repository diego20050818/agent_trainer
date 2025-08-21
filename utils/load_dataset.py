"""处理私有数据集的工具类
"""

from datasets import Dataset,load_dataset
from transformers import AutoTokenizer
from .logger import logger

class dataset_loder:
    """数据读取类
    """
    def __init__(self,dataset_path:str,
                 dataset_class:str,
                 max_length:int,
                 tokenizer,
                 **special_token:dict) -> None:
        """初始化数据读取

        Args:
            dataset_path (str): 数据集路径
            dataset_class (str): 数据集种类，json/csv
            max_length (int): 最大截断长度
            tokenizer (_type_): 模型分词器
        """
        
        logger.info("数据读取开始")
        self.dataset_path = dataset_path
        self.dataset_class = dataset_class

        self.maxlength = max_length
        self.tokenizer = tokenizer

        self.eos = special_token.get("eos")
        self.bos = special_token.get("bos")
        # self.think = special_token.get("think")


    def loder(self):
        """数据集加载器

        Returns:
            _type_: 返回dataset类
        """
        dataset = load_dataset(self.dataset_class,data_files=self.dataset_path)
        dataset = dataset['train']
        logger.info("数据下载完成")
        return dataset

    def process_qwen(self,example):
        """
        构造多轮对话数据的实例
        原理：直接构造包括多轮对话中所有机器人回复内容的标签，
        既充分地利用了所有机器人的回复信息，同时也不存在拆重复计算。

        inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
        labels = <-100> <assistant1> <-100> <assistant2> <-100> <assistant3>
        """
        max_length = self.maxlength
        BOS = self.bos
        EOS = self.eos
        tokenizer = self.tokenizer

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

    def process_gpt(self,example):
        """针对gpt-oss的多轮对话映射函数，注意这个函数并不会处理think块

        Args:
            example (_type_): 传入dataset.map函数默认
        """
        max_length = self.maxlength
        BOS = self.bos
        EOS = self.eos
        tokenizer = self.tokenizer

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

                    # 助手最终回复 -> 要作为 label
                    resp_text = f"{BOS}助手回复：{turn['content']}{EOS}"
                    resp_tokens = tokenizer.encode(resp_text, add_special_tokens=False)
                    input_ids.extend(resp_tokens)
                    labels.extend(resp_tokens)  # 学习助手回复

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

    
    def dataset_map(self,fuc,remove_columns:list) -> Dataset:
        """对数据集进行映射

        Args:
            fuc (fuction): 处理数据集的映射函数
            remove_columns (list): 需要删除的表头名称

        Returns:
            _type_: Dataset
        """
        dataset = self.loder()
        dataset = dataset.map(
            fuc,
            remove_columns=remove_columns
        )
        logger.info("数据映射完成")
        return dataset
