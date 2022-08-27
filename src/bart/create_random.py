import shutil
import random
import argparse
import os
from src.label.preprocess import check_fout, mkdir


def work(args):
    splits = ['train', 'test']
    for split in splits:
        # random src
        tgt_file_path = os.path.join(args.output_dir, f"{split}.source")
        fout = check_fout(tgt_file_path)
        src_dir = os.path.join(args.src_dir, f"{split}_src")
        data_num = len(os.listdir(src_dir))
        for i in range(data_num):
            src_txt_path = os.path.join(src_dir, f"{i}.txt")
            lines = open(src_txt_path, 'r').readlines()
            clean_lines = [line.strip() for line in lines]
            random.shuffle(clean_lines)
            fout.write(" ".join(clean_lines))
            fout.write("\n")
        # copy golden reference
        src_ref_file_path = os.path.join(args.gold_dir, f"{split}.target")
        tgt_ref_file_path = os.path.join(args.output_dir, f"{split}.target")
        shutil.copyfile(src_ref_file_path, tgt_ref_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bart model preprocess.')
    # S/L parameters
    parser.add_argument('--data_dir', type=str, default='/data/tsq/contrastive/clust_documents/')
    parser.add_argument('--category', type=str, choices=['animal', 'company', 'film', 'multi_news', 'wcep', 'copy'],
                        default='wcep')
    args = parser.parse_args()
    args.src_dir = os.path.join(args.data_dir, args.category, 'raw')
    args.output_dir = os.path.join(args.data_dir, args.category, 'bart', 'test_as_valid', 'random')
    if args.category == "wcep":
        args.gold_dir = os.path.join(args.data_dir, args.category, 'raw', 'origin')
    else:
        args.gold_dir = os.path.join(args.data_dir, args.category, 'bart', 'test_as_valid', 'tf_idf')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    work(args)
