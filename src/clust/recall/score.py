import argparse
import copy
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from src.bart.preprocess import work as preprocess
from src.bart.preprocess import check_fout
from src.clust.score import make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--rm_each_sent', action='store_true',
                        help='whether score paragraph and remove each sentence')
    parser.add_argument('--para_penal', type=float, default=0.5,
                        help='if use paragraph gain, the penalty of origin para ppl score')
    # data parameters
    parser.add_argument('--lead_section_num', type=int, default=10,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    parser.add_argument('--addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')
    # path

    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data1/tsq/contrastive/clust_documents",
                        help='dir of raw data (after clean noise)')
    parser.add_argument('--tgt_dir', type=str, default="/data1/tsq/contrastive/clust_documents/animal/extract",
                        help='dir where processed data will go')

    # Data parameters for regression/binary classify
    parser.add_argument('--task', default='regression', choices=['regression', 'binary'])
    parser.add_argument('--label', default='f1', choices=['recall', 'precision', 'f1', 'r2_recall'])
    parser.add_argument('--model', default='lgb', choices=['lgb', 'nn'])
    parser.add_argument('--train_num', type=int, default=500,
                        help='how many data for train split')

    args = parser.parse_args()

    print(args)
    return args


def work():
    args = parse_args()
    csv_dir = os.path.join(args.data_dir, args.category, 'ensemble', f'{args.label}_{args.task}',
                           f'few_shot{args.train_num}', args.model)

    args.prompt = f'{args.task}_{args.model}_{args.label}'
    origin_tgt_dir = os.path.join(args.data_dir, args.category, 'score', args.prompt)

    splits = ['train', 'test']
    for split in splits:
        # read csv
        csv_path = os.path.join(csv_dir, f'pred_{split}.csv')
        data_csv_table = pd.read_csv(csv_path)
        # output each score
        tgt_dir = os.path.join(origin_tgt_dir, f'{split}_src')

        if args.rm_each_sent:
            tgt_dir = os.path.join(tgt_dir, f'{args.prompt}_score_rm_each_sent')
        else:
            tgt_dir = os.path.join(tgt_dir, f'{args.prompt}_score')
        if not os.path.exists(tgt_dir):
            make_dirs(tgt_dir)

        data_id = -1
        # score_path = os.path.join(tgt_dir, f"{data_id}_score.txt")
        # fout = check_fout(score_path)
        for index, row in data_csv_table.iterrows():
            new_id = int(row['data_id'])
            if new_id != data_id:
                data_id = new_id
                score_path = os.path.join(tgt_dir, f"{data_id}_score.txt")
                fout = check_fout(score_path)
            fout.write(f"{row['pred_score']}\n")

        print("[2]Save scores at ", tgt_dir)


if __name__ == '__main__':
    work()
