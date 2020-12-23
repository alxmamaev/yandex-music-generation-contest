from data_utils import read_abc
from model import get_model
from train import get_training_files
from tqdm import tqdm
from pathlib import Path
import zipfile
import shutil
import requests
import torch
import youtokentome as yttm
from argparse import ArgumentParser
from math import ceil


def predict_notes(model, tokenizer, context_text):
    context = tokenizer.encode(context_text, bos=True, eos=True)
    if len(context) > 512:
        context_text = context_text.split(" @ ")
        context_text[1] = context_text[1].split(" | ")[1:]
        context_text[1] = " | ".join(context_text[1])
        context_text = " @ ".join(context_text)
        context = tokenizer.encode(context_text, bos=True, eos=True)


    context_tokens = torch.tensor(context, dtype=torch.long)
    context_tokens = context_tokens.unsqueeze(0).cuda()
    
    gen_tokens = model.generate(input_ids=context_tokens, 
                                max_length=300, 
                                min_length=32,
                                early_stopping=False,
                                num_beams=20,
                                bos_token_id=2, 
                                eos_token_id=3,
                                no_repeat_ngram_size=15,
                                pad_token_id=0,
                                bad_words_ids=[[114]])[0].tolist()

    notes = tokenizer.decode(gen_tokens, ignore_ids=[0,1,2,3])[0]
    notes = notes.replace(" ", "").replace("|", "|\n")
    
    return notes

def predict(model, tokenizer, text_path):
    contexts = read_abc(text_path)
    new_path = Path("predict_abc").joinpath(text_path.name)

    predicted_tokens = predict_notes(model, tokenizer, contexts)

    with open(text_path) as f:
        abc_text = f.read()

    with open(new_path, "w") as f:
        f.write(abc_text + predicted_tokens)
        

def parse():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=int)
    parser.add_argument("device_id", type=int)
    parser.add_argument("world_size", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    test_paths = get_training_files('testset/abc')
    test_paths = sorted(test_paths)

    part_size = ceil(len(test_paths) / args.world_size)
    test_paths = test_paths[args.device_id * part_size: (args.device_id + 1) * part_size]


    tokenizer = yttm.BPE("abc.yttm")
    model = get_model(tokenizer.vocab_size())
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.cuda()

    for p in tqdm(test_paths):
        midi_path = Path("predict_midi/").joinpath(p.name.replace(".abc", ".midi"))
        if midi_path.exists():
            continue
        
        abc_path = predict(model, tokenizer, p)