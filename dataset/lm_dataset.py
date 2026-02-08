import json
import os
import torch
from torch.utils.data import Dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dataset类

class PretrainDataset(Dataset):
    def __init__(self, datapath, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.sample = self.load_data(datapath)
        
    
    def load_data(self, datapath):
        samples = []
        with open(datapath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, idx):
        sample = self.sample[idx]

        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors = 'pt'
        )
        # (max_length,)
        input_ids = encoding['input_ids'].squeeze(0)

        loss_mask = input_ids!=self.tokenizer.pad_token_id


        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(input_ids[1:], dtype=torch.long)

        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.bool)

        return x, y, loss_mask