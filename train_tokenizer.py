from argparse import ArgumentParser
from data_utils import read_abc
from pathlib import Path
from tqdm import tqdm
import youtokentome as yttm

def parse():
    parser = ArgumentParser()
    parser.add_argument("datapath")
    parser.add_argument("model_path")
    parser.add_argument("--temp_corpus_path", default="train_corpus")
    parser.add_argument("--vocab_size", type=int, default=25000)

    return parser.parse_args()

def main(args):
    train_files = list(Path(args.datapath).glob("*.abc"))
    print("Creating temp corpus")
    with open(args.temp_corpus_path, "w") as f:
        for file in tqdm(train_files):
            (keys, notes) = read_abc(file)
            f.write(f"{keys}\n{notes}\n")

    print("Training model")
    yttm.BPE.train(data=args.temp_corpus_path, vocab_size=args.vocab_size, model=args.model_path)
    
    print("Finished")


if __name__ == "__main__":
    args = parse()
    main(args)