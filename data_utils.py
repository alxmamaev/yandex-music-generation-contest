import torch
from torch.nn.utils.rnn import pad_sequence

USEABLE_KEYS = ["M:", "L:", "Q:", "K:"]


def read_abc(path):
    keys = []
    notes = []
    with open(path) as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("%"):
                continue

            if any([line.startswith(key) for key in USEABLE_KEYS]):
                keys.append(line)
            else:
                notes.append(line)

    keys = " ".join(keys)
    notes = "".join(notes).strip()
    notes = notes.replace(" ", "")

    if notes.endswith("|"):
        notes = notes[:-1]

    notes = notes.replace("[", " [")
    notes = notes.replace("]", "] ")
    notes = notes.replace("(", " (")
    notes = notes.replace(")", ") ")
    notes = notes.replace("|", " | ")
    notes = notes.strip()
    notes = " ".join(notes.split())
    
    if not keys or not notes:
        return None

    text = keys + " @ " + notes

    return text


def collate_function(batch):
    features = [i["features"] for i in batch]
    target = [i["target"] for i in batch]
    
    features_lens = [len(i) for i in features]
    target_lens = [len(i) for i in target]
    
    max_features_len = max(features_lens)
    max_target_len = max(target_lens)
    
    features_mask = torch.tensor([[1] * l + [0] * (max_features_len - l) for l in features_lens],
                                 dtype=torch.bool)
    
    target_mask = torch.tensor([[1] * l + [0] * (max_target_len - l) for l in target_lens],
                                dtype=torch.bool)
    
    features_padded = pad_sequence(features, batch_first=True)
    target_padded = pad_sequence(target, batch_first=True)
    
    return {"input_ids": features_padded,
            "decoder_input_ids": target_padded,
            "labels": target_padded, 
            "attention_mask": features_mask, 
            "decoder_attention_mask": target_mask}