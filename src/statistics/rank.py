# coding: utf-8

import argparse
import os
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

from multiprocessing import Pool


def load_json_rank(oracle_result_dir):
    # read json oracle data
    _files = os.listdir(oracle_result_dir)
    json_file_num = 0
    for file in _files:
        if file[-4:] == 'json':
            json_file_num += 1

    candidates_list = []
    for json_id in tqdm(range(json_file_num), desc="load json"):
        json_file_path = os.path.join(oracle_result_dir, f"{json_id}_candidates.json")
        with open(json_file_path, "r") as fin:
            candidates = []
            lines = fin.readlines()
            for line in lines:
                candidate = json.loads(line)
                # candidate keys: "summary", "rouge_score", "rank"
                candidates.append(candidate)

        candidates_list.append(candidates)
    return candidates_list


def draw_line_chart(out_dir, pic_name: str, key_name: str, data_list: list):
    df = pd.DataFrame({key_name: data_list})
    sns.distplot(df, bins=20)
    plt.title(key_name, fontstyle='italic')
    plt.savefig(os.path.join(out_dir, f"{pic_name}.png"))


def count_candidate_num(candidates_list):
    candidate_num_list = [len(candidates) for candidates in candidates_list]

    return candidate_num_list


def get_origin_rank_percent_list(candidates_list):
    origin_rank_percent_list = []
    topic_num = len(candidates_list[0][0]["rank"])
    origin_combine = [0] * topic_num
    for candidates in candidates_list:
        candidate_num = len(candidates)
        origin_position = 0
        for iter, candidate in enumerate(candidates):
            if candidate["rank"] == origin_combine:
                origin_position = iter
                break
        # 100 is the highest rank_percent
        rank_percent = 100 * (candidate_num - origin_position) / candidate_num
        origin_rank_percent_list.append(rank_percent)

    return origin_rank_percent_list


def get_margin_list(candidates_list, simcls_result_sum_dir):
    rerank_margin_list = []
    topic_num = len(candidates_list[0][0]["rank"])
    origin_combine = [0] * topic_num
    try:
        for data_id, candidates in enumerate(candidates_list):
            simcls_sum_path = os.path.join(simcls_result_sum_dir, f"{data_id}_decoded.txt")
            simcls_res = open(simcls_sum_path, "r").read().strip()
            candidate_num = len(candidates)
            simcls_rouge = 0
            origin_rouge = 0
            for iter, candidate in enumerate(candidates):
                summary = candidate["summary"].strip()
                if summary == simcls_res:
                    simcls_rouge = candidate["rouge_score"]
                if candidate["rank"] == origin_combine:
                    origin_rouge = candidate["rouge_score"]

            rerank_margin_list.append(simcls_rouge - origin_rouge)
    except FileNotFoundError:
        # TODO there are 2 files missing in simcls_result_sum_dir
        pass

    return rerank_margin_list


def get_rerank_percent_list(candidates_list, simcls_result_sum_dir):
    rerank_list = []
    try:
        for data_id, candidates in enumerate(candidates_list):
            simcls_sum_path = os.path.join(simcls_result_sum_dir, f"{data_id}_decoded.txt")
            simcls_res = open(simcls_sum_path, "r").read().strip()
            candidate_num = len(candidates)
            origin_position = 0
            for iter, candidate in enumerate(candidates):
                if candidate["summary"].strip() == simcls_res:
                    origin_position = iter
                    break
            # 100 is the highest rank_percent
            rank_percent = 100 * (candidate_num - origin_position) / candidate_num
            rerank_list.append(rank_percent)
    except FileNotFoundError:
        # TODO there are 2 files missing in simcls_result_sum_dir
        pass

    return rerank_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        # default='/data1/tsq/contrastive/group/animal/clust3_combined/oracle/test/oracle_16candidates',
                        default='/data1/tsq/contrastive/group/animal/combine/fixed/clust3_combined/oracle/test/oracle_fix_t0_5candidates/candidates',
                        # default='/data1/tsq/contrastive/group/animal/bart_whole_combined/oracle/test/oracle_16candidates',
                        help="where the oracle results store")
    parser.add_argument('--simcls_result_sum_dir', type=str,
                        default='/data1/tsq/contrastive/group/animal/rerank/fixed/3centers_rawsrc/history/result_fix_t0_5candidates_721/21-07-20-31/sum',
                        help="where the rerank result is")
    parser.add_argument('--out_root_dir', type=str,
                        default='/home/tsq/TopCLS/src/statistics/pictures',
                        help="where the picture will go")
    parser.add_argument('--method', type=str, default="group_v2_fixed",
                        choices=["group_v2_fixed", "group_v1", "group_clean", "bart_whole", "bart_kmeans"])
    parser.add_argument('--pic_name', type=str, default="margin",
                        choices=["candidate_num", "origin_rank_percent", "rerank_rank_percent", "margin"])
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_root_dir, args.method)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    candidates_list = load_json_rank(args.data_dir)

    if args.pic_name == "candidate_num":
        # candidate_num is # of combined candidate summary
        candidate_num_list = count_candidate_num(candidates_list)
        draw_line_chart(args.out_dir, "candidate_num", "candidate_num of each summary", candidate_num_list)
    elif args.pic_name == "origin_rank_percent":
        # origin_rank_percent is the top rank percent of [0,0,0] candidate.
        # If origin_rank_percent is 100, that means the [0,0,0] is the combine that gets highest rouge
        origin_rank_percent_list = get_origin_rank_percent_list(candidates_list)
        draw_line_chart(args.out_dir, "origin_rank_percent", "top rank percent of origin beam search candidate",
                        origin_rank_percent_list)
    elif args.pic_name == "rerank_rank_percent":
        rerank_rank_percent_list = get_rerank_percent_list(candidates_list, args.simcls_result_sum_dir)
        print("rank 1st: ", rerank_rank_percent_list.count(100))
        print("data num: ", len(rerank_rank_percent_list))
        draw_line_chart(args.out_dir, "rerank_rank_percent", "top rank percent of simcls rerank candidate",
                        rerank_rank_percent_list)
    elif args.pic_name == "margin":
        rerank_margin_list = get_margin_list(candidates_list, args.simcls_result_sum_dir)
        print(f"Total gain: {sum(rerank_margin_list)}")
        print(f"Average gain: {sum(rerank_margin_list) / len(rerank_margin_list)}")
        draw_line_chart(args.out_dir, "margin", "simcls rerank result rouge minus origin rouge",
                        rerank_margin_list)
