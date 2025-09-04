from load_data import load_fold
from tqdm import tqdm
import argparse


def save_dataset_to_txt(dataset, output_file):
    with open(output_file, "w") as f:
        for idx in tqdm(range(len(dataset))):
            image, label = dataset[idx]
            image_path = dataset.image_paths[idx]  # 获取对应的路径
            f.write(f"{image_path} {label}\n")


def merge_txt_files(input_files, output_file):
    with open(output_file, "w") as outfile:
        for input_file in tqdm(input_files):
            with open(input_file, "r") as infile:
                for line in infile:
                    outfile.write(line)


def main(args):
    if args.todo == "save":
        path = args.paths[0]
        out_path = args.out_path
        train_dataset, val_dataset = load_fold(path=path)

        save_dataset_to_txt(train_dataset, out_path + "_train.txt")
        save_dataset_to_txt(val_dataset, out_path + "_test.txt")

    elif args.todo == "merge":
        paths = args.paths
        out_path = args.out_path
        train_paths = [path + "_train.txt" for path in paths]
        test_paths = [path + "_test.txt" for path in paths]
        merge_txt_files(train_paths, out_path + "_train.txt")
        merge_txt_files(test_paths, out_path + "_test.txt")


if __name__ == "__main__":
    conf = argparse.ArgumentParser()
    conf.add_argument(
        "--paths", type=str, nargs="+", default=None, help="Paths to the datasets"
    )
    conf.add_argument("--out_path", default=None, type=str)
    conf.add_argument("--todo", default=None, type=str)

    args = conf.parse_args()
    main(args)
