from argparse import ArgumentParser
from Levenshtein import distance
from data_utils import read_abc
from glob import glob
from tqdm import tqdm
from pathlib import Path


def bars_similiarity(bar1, bar2):
    distances = []
    for n1 in bar1:
        distances.append(min([distance(n1, n2) / (len(n1) + len(n2)) for n2 in bar2]))
    
    return sum(distances) / len(distances)


def parse():
    parser = ArgumentParser()

    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    return parser.parse_args()


def main(args):
    bars = []
    bars_similaruty = []

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    file_index = 0

    print("preprocessing data...")
    for i in tqdm(list(input_dir.glob("*.abc"))):
        abc = read_abc(i)
        if abc is None:
            continue
            
        keys, abc = abc.split(" @ ")
        abc = abc.replace(" ", "").split("|")
        num_bars = len(abc) // 16
        
        if num_bars == 0:
            continue
            
            
        for i in range(num_bars):
            bar1 = abc[i * 8 : (i + 1) * 8]
            bar2 = abc[(i + 1) * 8 : (i + 2) * 8]

            if len(bar1) + len(bar2) != 16:
                continue

            if "x8" in "|\n".join(bar1 + bar2):
                continue

            sim = bars_similiarity(bar1, bar2)
            
            if sim < 0.43:
                continue

            with open(output_dir.joinpath(f"{file_index}.abc"), "w") as f:
                new_abc = keys.replace(" ", "\n") + "\n" + "|\n".join(bar1 + bar2)
                f.write(new_abc)
            
            file_index += 1

if __name__ == "__main__":
    args = parse()
    main(args)