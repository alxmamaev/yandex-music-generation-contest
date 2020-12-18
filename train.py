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
    parser.add_argument("--min_tokens_in_bar", default=1, type=int)
    parser.add_argument("--max_tokens_in_bar", default=37, type=int)
    parser.add_argument('--check', action='store_true')

    return parser.parse_args()


def get_training_files(train_dir):
    train_dir = Path(train_dir)

    return list(train_dir.glob("*.abc"))


def main(args):
    training_args = TrainingArguments(
        output_dir="./ABCModel",
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
    train_texts = [read_abc(p) for p in train_paths]

    print("Making dataset...")
    train_dataset = ABCDataset(train_texts, tokenizer,
                            min_tokens_in_bar=args.min_tokens_in_bar,
                            max_tokens_in_bar=args.max_tokens_in_bar)

    for i in range(100):
        item = train_dataset[0]

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

