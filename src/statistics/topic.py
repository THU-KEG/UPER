import json
import os
from tqdm import tqdm
import argparse
import torch
from transformers import BertTokenizer, RobertaTokenizer, BertForMaskedLM, BertConfig
from src.clust.pattern import select_clusts
from src.clust.gather import check_fout


def get_avg(doc_statistics_dict, total_doc_num):
    for k, v in doc_statistics_dict.items():
        if isinstance(v, list):
            avg_v = [topic_v / total_doc_num for topic_v in v]
        else:
            avg_v = v / total_doc_num
        doc_statistics_dict[k] = avg_v
    return doc_statistics_dict


def run(args):
    # get input
    save_root = os.path.join(args.data_dir, args.category, 'classify', args.split)
    if args.object == 'tgt':
        save_root = os.path.join(save_root, 'tgt')
    if args.last_noisy:
        topic_dir = os.path.join(save_root, f'ls{args.lead_section_num}_t{args.topic_num}_noise')
    else:
        topic_dir = os.path.join(save_root, f'ls{args.lead_section_num}_t{args.topic_num}')
    if args.max_logit:
        topic_dir = os.path.join(topic_dir, 'max_logit')

    # get id path
    id_root = os.path.join(args.data_dir, args.category, 'result', f'inverse_add{args.addition_pattern_num}',
                           f'{args.split}_{args.object}', 'inverse_score', 'ws_0.75', 'zs_classify')
    if args.last_noisy:
        result_dir = os.path.join(id_root,
                                  f'k{args.topic_num}_i{args.clust_input_sent_num}_o{args.clust_output_sent_num}_{args.proportion}_noise')
    elif args.max_logit:
        result_dir = os.path.join(id_root,
                                  f'k{args.topic_num}_i{args.clust_input_sent_num}_o{args.clust_output_sent_num}_{args.proportion}_max')
    else:
        result_dir = os.path.join(id_root,
                                  f'k{args.topic_num}_i{args.clust_input_sent_num}_o{args.clust_output_sent_num}_{args.proportion}')

    id_dir = os.path.join(result_dir, 'top_bottom')

    # read topic feature logits for each document
    # each list in statistics_dict should be the length of topic_num
    statistics_dict = {
        "hard_topic": [0] * args.topic_num,
        "soft_topic": [0] * args.topic_num,
    }
    total_doc_num = 0
    data_num = len(os.listdir(topic_dir))
    if args.use_subset:
        assert data_num == len(os.listdir(id_dir))
    for data_id in tqdm(range(data_num)):
        topic_path = os.path.join(topic_dir, f"{data_id}_class.json")
        topic_lines = open(topic_path, 'r').readlines()
        if args.use_subset:
            id_path = os.path.join(id_dir, f"{data_id}_id.txt")
            id_lines = open(id_path, 'r').readlines()
            ids = [int(id_line.strip()) for id_line in id_lines]
            doc_num = len(ids)
        else:
            doc_num = len(topic_lines)
        total_doc_num += doc_num
        doc_statistics_dict = {
            "hard_topic": [0] * args.topic_num,
            "soft_topic": [0] * args.topic_num,
        }
        for doc_id in range(0, doc_num):
            if args.use_subset:
                try:
                    doc_id = ids[doc_id]
                except IndexError:
                    # no enough docs
                    break
            class_dict = json.loads(topic_lines[doc_id].strip())
            assert class_dict["doc_id"] == doc_id
            topic = class_dict["topic"]
            scores = class_dict["scores"]
            statistics_dict["hard_topic"][topic] += 1
            for topic_id, score in enumerate(scores):
                statistics_dict["soft_topic"][topic_id] += score
            # class_dict = {"doc_id": doc_id + i, "topic": topic_num, "scores": topic_scores}
    avg_doc_statistics_dict = get_avg(statistics_dict, total_doc_num)
    print(args)
    print(avg_doc_statistics_dict)


if __name__ == '__main__':
    # test_data_generator()
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='/data1/tsq/contrastive/clust_documents/')
    parser.add_argument('--split', type=str, default="test", choices=['train', 'valid', 'test'],
                        help='data split')
    parser.add_argument('--object', default='tgt', choices=['src', 'tgt'])

    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])

    # data parameters
    parser.add_argument('--lead_section_num', type=int, default=20,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    parser.add_argument('--last_noisy', action='store_true', help='whether use an extra topic as noise')
    parser.add_argument('--max_logit', action='store_true', help='whether use max or avg logit')
    parser.add_argument('--addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')
    parser.add_argument('--clust_num', type=int, default=4,
                        help='how many cluster')
    parser.add_argument('--clust_input_sent_num', type=int, default=128,
                        help='how many sentences that can participate in clustering')
    parser.add_argument('--clust_output_sent_num', type=int, default=64,
                        help='if sent_num < clust_output_sent_num, we output them all and do not perform clustering')
    parser.add_argument('--proportion', type=str, default="free",
                        choices=['free', 'tp'],
                        help='how to decide the proportion of different topics')

    parser.add_argument('--use_subset', action='store_true', help='whether use extracted subset to count topic')

    args = parser.parse_args()
    run(args)
