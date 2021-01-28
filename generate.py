from data_utils import read_abc
from model import get_model
from train import get_training_files
from tqdm import tqdm
from pathlib import Path
import torch
import youtokentome as yttm
from argparse import ArgumentParser


def predict_notes(model, tokenizer, keys, notes):
    keys_tokens = tokenizer.encode(keys)
    notes_tokens = tokenizer.encode(notes)

    # TODO fix max lenght of transformer
    if len(keys_tokens) + len(notes_tokens) > 510:
        notes_tokens = notes_tokens[len(notes_tokens) - len(keys_tokens) - 510:]

    context_tokens = [2] + keys_tokens + notes_tokens + [3]

    context_tokens = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)

    if torch.cuda.is_available():
        context_tokens = context_tokens.cuda()
    
    bad_words_ids = []
    bad_words = ["x8 | "]
    for w in bad_words:
        bad_words_ids.append(tokenizer.encode(bad_words)[0])

    gen_tokens = model.generate(input_ids=context_tokens, 
                                max_length=320, 
                                min_length=32,
                                early_stopping=False,
                                num_beams=20,
                                bos_token_id=2, 
                                eos_token_id=3,
                                no_repeat_ngram_size=15,
                                pad_token_id=0,
                                bad_words_ids=bad_words_ids)
                                
    gen_tokens = gen_tokens[0].tolist()

    notes = tokenizer.decode(gen_tokens, ignore_ids=[0,1,2,3])[0]
    notes = notes.replace(" ", "").replace("|", "|\n")
    
    return notes

def predict(model, tokenizer, text_path, output_dir):
    keys, notes = read_abc(text_path)
    new_path = output_dir.joinpath(text_path.name)

    predicted_tokens = predict_notes(model, tokenizer, keys, notes)

    with open(text_path) as f:
        abc_text = f.read()

    with open(new_path, "w") as f:
        f.write(abc_text + predicted_tokens)

    return new_path
        

def parse():
    parser = ArgumentParser()
    parser.add_argument("datapath", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--tokenzer", default="abc.yttm")
    parser.add_argument("--output_dir", default="predict_abc")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    test_paths = get_training_files(args.datapath)
    test_paths = sorted(test_paths)

    tokenizer = yttm.BPE(args.tokenzer)
    model = get_model(tokenizer.vocab_size())
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()

    print("Starts generation")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    for p in tqdm(test_paths):
        abc_path = predict(model, tokenizer, p, output_dir)
        print(abc_path)