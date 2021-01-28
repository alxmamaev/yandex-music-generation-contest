import random
import torch
from torch.utils.data import Dataset


class ABCDataset(Dataset):
    def __init__(self, data, 
                 context_bars_num=8, 
                 target_bars_num=8,
                 bos_id=2,
                 eos_id=3,
                 is_test=False):
        
        self.notes = []
        self.keys = []

        for (keys, notes) in data:
            if notes is None:
                continue

            self.keys.append(keys)
            self.notes.append(notes)
        
        self.context_bars_num = context_bars_num
        self.target_bars_num = target_bars_num
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.is_test = is_test
        
    def __len__(self):
        return len(self.keys)
    
    
    def __getitem__(self, idx):
        notes = self.notes[idx]
        keys = self.keys[idx]
        
        if not self.is_test:
            split_indx = 8

            # split notes to context (input for network) and target (that model must to generate)
            context_notes = notes[split_indx - self.context_bars_num : split_indx]
            target_notes = notes[split_indx: split_indx + self.target_bars_num]
        else:
            context_notes = notes
            target_notes = []

        context_tokens = [self.bos_id] + keys
        target_tokens = [self.bos_id]

        for bar in context_notes:
            context_tokens += bar

        for bar in target_notes:
            target_tokens += bar

        context_tokens += [self.eos_id]
        target_tokens += [self.eos_id]

        context_tokens = torch.tensor(context_tokens, dtype=torch.long)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return {"features": context_tokens, "target": target_tokens}