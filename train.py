import torch
from tqdm import tqdm
from argparse import ArgumentParser
from model import get_model
from data_utils import read_abc, collate_function
from dataset import ABCDataset
import youtokentome as yttm
from transformers import Trainer, TrainingArguments
from pathlib import Path


def parse():
    parser = ArgumentParser()
    parser.add_argument("train_dir")
    parser.add_argument("--tokenizer", default="abc.yttm")
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--save_steps", default=10, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--n_workers", default=0, type=int)
    parser.add_argument("--min_sequence_lenght", default=16, type=int)
    parser.add_argument("--max_sequence_lenght", default=512, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output_dir", default="./ABCModel", type=str)
    parser.add_argument('--check', action='store_true')

    return parser.parse_args()


def get_training_files(dir):
    dir = Path(dir)

    return list(dir.glob("*.abc"))


def main(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=10,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.n_workers,
    )

    print("Loading tokenizer...")
    tokenizer = yttm.BPE(args.tokenizer)

    print("Loading model...")
    model = get_model(vocab_size=tokenizer.vocab_size())

    print("List training files...")
    train_paths = get_training_files(args.train_dir)

    if args.check:
        train_paths = train_paths[:10000]

    print("Loading train texts...")
    train_data = []
    for p in tqdm(train_paths):
        (keys, notes) = read_abc(p)
        if keys is None:
            continue
        
        keys_tokens = tokenizer.encode(keys)
        bars = notes.split(" | ")
        notes_tokens = [tokenizer.encode(i + " | ") for i in bars]

        ## To avoid OOM
        sequence_len = sum(len(i) for i in notes_tokens)
        if not (args.min_sequence_lenght < sequence_len < args.max_sequence_lenght):
            print("Skip", p)
            continue

        train_data.append((keys_tokens, notes_tokens))
        

    print("Making dataset...")
    train_dataset = ABCDataset(train_data)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_function,
        train_dataset=train_dataset
    )

    print("Start training...")
    trainer.train()


if __name__ == "__main__":
    args = parse()
    main(args)

