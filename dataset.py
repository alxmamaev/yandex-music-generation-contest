import random
import torch
from torch.utils.data import Dataset


class ABCDataset(Dataset):
    def __init__(self, texts, tokenizer, 
                 context_bars_num=8, 
                 target_bars_num=8,
                 min_tokens_in_bar=1,
                 max_tokens_in_bar=37,
                 is_test=False):
        
        self.notes = []
        self.keys = []

        for text in texts:
            if text is None:
                continue

            text = text.strip()
            keys, notes = text.split("@")
            notes = notes.split(" | ")

            keys = tokenizer.encode(keys)
            notes = [tokenizer.encode(i) for i in notes]
            
            if not is_test:
                if len(notes) < context_bars_num + target_bars_num:
                    continue

                num_tokens = [len(i) for i in notes]
                if max_tokens_in_bar is not None and max(num_tokens) > max_tokens_in_bar:
                    continue

                if min_tokens_in_bar is not None and min(num_tokens) < min_tokens_in_bar:
                    continue

            self.keys.append(keys)
            self.notes.append(notes)
        
        self.tokenizer = tokenizer
        self.context_bars_num = context_bars_num
        self.target_bars_num = target_bars_num
        self.is_test = is_test

        self.bos = 2
        self.eos = 3
        self.flag_token = tokenizer.encode(" @ ")[0]
        self.bar_token = tokenizer.encode(" | ")[0]
        
    def __len__(self):
        return len(self.keys)
    
    
    def __getitem__(self, idx):
        notes = self.notes[idx]
        keys = self.keys[idx]
        
        if not self.is_test:
            split_indx = random.randint(self.context_bars_num, len(notes) - self.target_bars_num)

            context_notes = notes[split_indx - self.context_bars_num : split_indx]
            target = notes[split_indx: split_indx + self.target_bars_num]

            if split_indx + self.target_bars_num == len(notes):
                context_notes.append([self.flag_token]) # to learn ends of soungs
        else:
            context_notes = notes
            target = []

        context_tokens = [self.bos] + keys + [self.flag_token]
        
        for i in context_notes:
            context_tokens += i + [self.bar_token]

        context_tokens.append(self.eos)

        target_tokens = [self.bos]
        for i in target:
            target_tokens += i + [self.bar_token]

        target_tokens.append(self.eos)

        context_tokens = torch.tensor(context_tokens, dtype=torch.long)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return {"features": context_tokens, "target": target_tokens}