import json
import os
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info
from itertools import cycle

class PretrainDataset(IterableDataset):
    def __init__(self, data_paths, tokenizer, max_length=512, probabilities=None):
        """
        :param data_paths: list, 包含多个jsonl文件的路径列表
        :param probabilities: list, 对应每个文件的采样概率 (e.g., [0.7, 0.3])。如果为None则均匀采样。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 确保传入的是列表
        if isinstance(data_paths, str):
            self.data_paths = [data_paths]
        else:
            self.data_paths = data_paths
            
        self.probabilities = probabilities
        
        # 如果没有指定概率，则根据文件大小粗略估算或均匀分配，这里默认均匀分配
        if self.probabilities is None:
            self.probabilities = [1.0 / len(self.data_paths)] * len(self.data_paths)
            
        assert len(self.data_paths) == len(self.probabilities), "文件数量必须与概率数量一致"

    def read_file_stream(self, path):
        """生成器：流式读取单个文件"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    yield data
                except json.JSONDecodeError:
                    continue

    def process_sample(self, sample):
        """将文本转换为模型输入格式"""
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # 【关键1】关闭自动添加，手动控制
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)  # 1=有效token, 0=padding
        
        # 【关键2】使用 attention_mask 准确定位真实数据的结尾
        # 这比查找 pad_token_id 更可靠，避免文本中本身就有该 token 的歧义
        valid_length = attention_mask.sum().item()  # 真实token数量
        
        # 【关键3】在真实数据后手动添加 <|endoftext|> (ID: 151643)
        if valid_length < self.max_length:
            # 有空间：在真实数据后追加 EOS
            input_ids[valid_length] = 151643
        else:
            # 满了：替换最后一个 token 为 EOS（确保序列以 EOS 结尾）
            input_ids[self.max_length - 1] = 151643
        
        # 【关键4】生成 loss_mask
        # 注意：EOS 也应该参与 loss 计算（模型需要学会生成 EOS）
        loss_mask = input_ids != self.tokenizer.pad_token_id

        # 【关键5】生成训练对 (input, target)
        # x: [tok1, tok2, ..., tokN-1]  预测下一个
        # y: [tok2, tok3, ..., tokN]    目标序列（包含EOS）
        x = input_ids[:-1].clone().detach().long()
        y = input_ids[1:].clone().detach().long()
        loss_mask = loss_mask[1:].clone().detach().bool()

        return x, y, loss_mask

    def __iter__(self):
        worker_info = get_worker_info()
        # 创建每个文件的迭代器
        iterators = [cycle(self.read_file_stream(path)) for path in self.data_paths]
        
        # 如果是多worker环境，需要做分片处理（这里简化处理，每个worker都读取所有数据但通过随机性错开）
        # 更加严谨的做法是在 read_file_stream 里根据 worker_id 跳过数据
        
        while True:
            # 根据概率随机选择一个数据源
            chosen_idx = random.choices(range(len(self.data_paths)), weights=self.probabilities, k=1)[0]
            try:
                # 获取该数据源的下一个样本
                sample = next(iterators[chosen_idx])
                yield self.process_sample(sample)
            except StopIteration:
                # 理论上使用了 cycle 不会停止，除非文件为空
                continue