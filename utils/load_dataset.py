"""处理私有数据集的工具类
"""
from typing import Optional
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
                 eval_dataset_path:Optional[str]=None,
                 **special_token:dict) -> None:
        """数据读取初始化参数

        Args:
            dataset_path (str): 训练数据集路径
            dataset_class (str): 数据集格式-json/csv
            max_length (int): 最大截断长度
            tokenizer (_type_): 模型分词器
            eval_dataset_path (Optional[str], optional): 验证数据集路径. Defaults to None.
        """
        
        logger.info("数据读取开始")
        self.dataset_path = dataset_path
        self.eval_dataset_path = eval_dataset_path if eval_dataset_path is not None else dataset_path
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
        
        if self.eval_dataset_path is not None:
            eval_dataset = load_dataset(self.dataset_class,data_files=self.eval_dataset_path)
            eval_dataset = eval_dataset['train']
        
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

    def process_qwen_Psy(self,example):
        """
        构造多轮对话数据的实例
        针对心理问答数据集做的优化：心理咨询师数字孪生（SoulChat2.0）
        原理：直接构造包括多轮对话中所有机器人回复内容的标签，
        既充分地利用了所有机器人的回复信息，同时也不存在拆重复计算。

        inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
        labels = <-100> <assistant1> <-100> <assistant2> <-100> <assistant3>
        """
        max_length = self.maxlength
        BOS = self.bos
        EOS = self.eos
        tokenizer = self.tokenizer

        system_prompt = """
        你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，
        能够合理地采用理情行为疗法给来访者提供专业地指导和支持，
        缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。
        理情行为治疗主要包括以下几个阶段，下面是对话阶段列表，
        并简要描述了各个阶段的重点。
        \n（1）**检查非理性信念和自我挫败式思维**：
        理情行为疗法把认知干预视为治疗的“生命”，
        因此，几乎从治疗一开始，在问题探索阶段，
        咨询师就以积极的、说服教导式的态度帮助来访者探查隐藏在情绪困扰后面的原因，
        包括来访者理解事件的思维逻辑，产生情绪的前因后果，
        借此来明确问题的所在。
        咨询师坚定地激励来访者去反省自己在遭遇刺激事件后，
        在感到焦虑、抑郁或愤怒前对自己“说”了些什么。\n
        （2）**与非理性信念辩论**：咨询师运用多种技术（主要是认知技术）帮助来访者向非理性信念和思维质疑发难，
        证明它们的不现实、不合理之处，认识它们的危害进而产生放弃这些不合理信念的愿望和行为。
        \n（3）**得出合理信念，学会理性思维**：在识别并驳倒非理性信念的基础上，
        咨询师进一步诱导、帮助来访者找出对于刺激情境和事件的适宜的、理性的反应，
        找出理性的信念和实事求是的、指向问题解决的思维陈述，
        以此来替代非理性信念和自我挫败式思维。为了巩固理性信念，
        咨询师要向来访者反复教导，证明为什么理性信念是合情合理的，
        它与非理性信念有什么不同，为什么非理性信念导致情绪失调，
        而理性信念导致较积极、健康的结果。
        \n（4）**迁移应用治疗收获**：积极鼓励来访者把在治疗中所学到的客观现实的态度，
        科学合理的思维方式内化成个人的生活态度，并在以后的生活中坚持不懈地按理情行为疗法的教导来解决新的问题。
        目前的场景为：{}
        """

        system_prompt = system_prompt.format(example['normalizedTag'])
        conversation_pair = example['messages']

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

    
    def dataset_map(self,fuc) -> Dataset:
        """对数据集进行映射

        Args:
            fuc (fuction): 处理数据集的映射函数

        Returns:
            _type_: Dataset
        """
        dataset = self.loder()
        currect_columns = ['input_ids','attention_mask','labels']
        columns = [i for i in list(dataset.column_names) if i not in currect_columns]
        dataset = dataset.map(
            fuc,
            remove_columns=columns
        )
        logger.info("数据映射完成")
        return dataset
