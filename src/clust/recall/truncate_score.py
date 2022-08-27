import argparse
import shutil
import pickle
from tqdm import tqdm
import os
import json
from src.clust.gather import check_fout


def make_dirs(dir):
    if not (os.path.exists(dir)):
        os.makedirs(dir)


def work():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="inverse",
                        choices=['qa', 'inverse', 'none', 'rouge',
                                 'regression_lgb', 'regression_nn',
                                 'regression_lgb_r2_recall', 'regression_nn_r2_recall'],
                        help='ways of prompt')
    parser.add_argument('--split', type=str, default="train", choices=['train', 'valid', 'test'],
                        help='data split')
    parser.add_argument('--level', type=str, default="all", choices=['all', 'topic', 'lead_section',
                                                                     'recall', 'precision', 'f1', 'r1_recall',
                                                                     'r2_recall', 'rl_recall'],
                        help='level of sort, first three are choices for qa prompt, last three are for rouge')
    parser.add_argument('--origin_addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')
    parser.add_argument('--new_addition_pattern_num', type=int, default=0,
                        help='# of additional patterns for inverse prompt')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data1/tsq/contrastive/clust_documents",
                        help='dir of raw data (after clean noise)')
    args = parser.parse_args()

    args.sentences_dir = os.path.join(args.data_dir, args.category, 'raw')
    args.classify_dir = os.path.join(args.data_dir, args.category, 'classify')
    if args.prompt == 'inverse':
        pattern_dir = os.path.join(args.data_dir, args.category, 'patterns',
                                   f'inverse_add{args.origin_addition_pattern_num}')
        score_dir = os.path.join(args.data_dir, args.category, 'score',
                                 f'inverse_add{args.origin_addition_pattern_num}', f"{args.split}_{args.object}",
                                 "inverse_score")
        new_pattern_dir = os.path.join(args.data_dir, args.category, 'patterns',
                                       f'inverse_add{args.new_addition_pattern_num}')
        new_score_dir = os.path.join(args.data_dir, args.category, 'score',
                                     f'inverse_add{args.new_addition_pattern_num}', f"{args.split}_{args.object}",
                                     "inverse_score")
        make_dirs(new_pattern_dir)
        make_dirs(new_score_dir)

        data_num = len(os.listdir(score_dir))
        for data_id in tqdm(range(data_num)):
            score_file = os.path.join(score_dir, f"{data_id}_score.txt")
            new_score_file = os.path.join(new_score_dir, f"{data_id}_score.txt")
            with open(score_file, 'r') as fin:
                fout = check_fout(new_score_file)
                lines = fin.readlines()
                for line in lines:
                    score_list = json.loads(line.strip())
                    new_score_list = score_list[:args.new_addition_pattern_num + 1]
                    fout.write(json.dumps(new_score_list))
                    fout.write('\n')


if __name__ == '__main__':
    work()
