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


def get_score_path(args, prompt, split):
    if prompt == 'inverse':
        tgt_dir = os.path.join(args.data_dir, args.category, 'score',
                               f'inverse_add{args.addition_pattern_num}', f"{split}_{args.object}")

    elif prompt == 'rouge':
        # this is not a typical prompt, we just need rouge score as a upper bound
        tgt_dir = os.path.join(args.data_dir, args.category, 'score', 'rouge', f"{split}_{args.object}")

    else:
        tgt_dir = os.path.join(args.data_dir, args.category, 'score',
                               f'tn{args.topic_num}_lsn{args.lead_section_num}', f"{split}_{args.object}")
    if args.rm_each_sent:
        tgt_dir = os.path.join(tgt_dir, f'{prompt}_para_penal{args.para_penal}')
    else:
        tgt_dir = os.path.join(tgt_dir, f'{prompt}_score')

    return tgt_dir


def get_score_dict(level, scores_list, prompt):
    score_dict = {}
    if prompt == 'qa':
        # lead section
        _score_list = []
        for i in range(len(scores_list)):
            sec_num = len(scores_list[i])
            topic_score_list = [scores_list[i][j] for j in range(sec_num)]
            _score_list.extend(topic_score_list)
        for i, score in enumerate(_score_list):
            score_dict[f'topic_sec{i}'] = score
    elif prompt == 'inverse':
        # inverse
        for i, score in enumerate(scores_list):
            score_dict[f'inverse{i}'] = score
    elif prompt == 'rouge':
        if level == 'recall':
            new_scores_list = [scores_list[0], scores_list[3], scores_list[6]]
        elif level == 'r2_recall':
            new_scores_list = [scores_list[3]]
        elif level == 'precision':
            new_scores_list = [scores_list[1], scores_list[4], scores_list[7]]
        else:  # f1
            new_scores_list = [scores_list[2], scores_list[5], scores_list[8]]
        score_dict[f'label'] = sum(new_scores_list) / len(new_scores_list)
    else:
        # none prompt
        score_dict['none_prompt'] = scores_list[0]
    return score_dict


def get_data_dicts(rouge_score_dir, split, label, train_num, prompt):
    if split == 'train':
        end_id = train_num
    else:
        end_id = len(os.listdir(rouge_score_dir))
    data_dicts = []
    for data_id in range(0, end_id):
        scores_path = os.path.join(rouge_score_dir, f'{data_id}_score.txt')
        score_lines = open(scores_path, 'r').readlines()
        data2sent2score_dict = []
        for sent_id, score_line in enumerate(score_lines):
            sent2score_dict = {'data_id': data_id, 'sent_id': sent_id, 'split': split}
            if prompt == 'qa' or prompt == 'inverse' or prompt == 'rouge':
                # 'qa' key: sent_id, value: scores_list [[float]] (i: topic_id j: lead_section_id)
                # 'inverse' key: sent_id, value: scores_list [float] (i: pattern_id)
                scores_list = json.loads(score_line.strip())
                score_dict = get_score_dict(label, scores_list, prompt)

            else:
                # prompt is none
                score_dict = get_score_dict(label, [float(score_line.strip())], prompt)
            sent2score_dict.update(score_dict)
            data2sent2score_dict.append(sent2score_dict)

        data_dicts.append(data2sent2score_dict)

    print(f"[1] Read {end_id} data from {rouge_score_dir}")
    return data_dicts


def merge_dict(rouge_data_dicts, data_dicts):
    for data_id in range(len(rouge_data_dicts)):
        data2sent2score_dict = rouge_data_dicts[data_id]
        for sent_id, rouge_score_dict in enumerate(data2sent2score_dict):
            _score_dict = data_dicts[data_id][sent_id]
            assert _score_dict['data_id'] == rouge_score_dict['data_id']
            assert _score_dict['sent_id'] == rouge_score_dict['sent_id']
            rouge_score_dict.update(_score_dict)
            rouge_data_dicts[data_id][sent_id] = rouge_score_dict

    return rouge_data_dicts


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
    parser.add_argument('--train_num', type=int, default=500,
                        help='how many data for train split')

    args = parser.parse_args()

    print(args)
    return args


def work():
    args = parse_args()
    tgt_dir = os.path.join(args.data_dir, args.category, 'ensemble', f'{args.label}_{args.task}',
                           f'few_shot{args.train_num}')

    if not os.path.exists(tgt_dir):
        make_dirs(tgt_dir)

    splits = ['train', 'test']
    prompts = ['qa', 'inverse', 'none']
    for split in splits:
        # get labels
        rouge_score_path = get_score_path(args, 'rouge', split)
        rouge_data_dicts = get_data_dicts(rouge_score_path, split, args.label, args.train_num, 'rouge')

        # get features
        for prompt in prompts:
            score_path = get_score_path(args, prompt, split)
            data_dicts = get_data_dicts(score_path, split, args.label, args.train_num, prompt)
            rouge_data_dicts = merge_dict(rouge_data_dicts, data_dicts)

        # init the csv dict
        final_dict = {k: [] for k in rouge_data_dicts[0][0].keys()}
        for data_id in range(len(rouge_data_dicts)):
            data2sent2score_dict = rouge_data_dicts[data_id]
            for sent_id, rouge_score_dict in enumerate(data2sent2score_dict):
                for k, v in rouge_score_dict.items():
                    final_dict[k].append(v)
        # transform to csv
        data_csv_table = pd.DataFrame.from_dict(final_dict)
        data_csv_table.set_index(['data_id', 'sent_id'])
        csv_path = os.path.join(tgt_dir, f'{split}.csv')
        data_csv_table.to_csv(csv_path)
        print("[2]Save csv at ", csv_path)


if __name__ == '__main__':
    work()
