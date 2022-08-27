# coding: utf-8

import argparse
import os
import json
import logging

logging.basicConfig(level=logging.INFO)

from multiprocessing import Pool


def line_count_src(line):
    SEP = '<EOP>'
    sents = line.strip().split(SEP)
    sent_num = len(sents)
    token_num = sum([len(sent.split()) for sent in sents])

    return sent_num, token_num


def line_count_tgt(line):
    SEP = '<SNT>'
    sents = line.strip().split(SEP)
    sent_num = len(sents)
    token_num = sum([len(sent.split()) for sent in sents])

    return sent_num, token_num


def check_align(args):
    data_dir = os.path.join(args.data_dir)
    splits = ['train', 'valid', 'test']
    results = {}
    for split in splits:
        with open(os.path.join(data_dir, f"{split}.src")) as src, open(
                os.path.join(data_dir, f"{split}.tgt")) as tgt:
            src_lines = list(src.readlines())
            print(f"{split}.src data_num: {len(src_lines)}")
            with Pool(processes=8) as pool:
                results_src = pool.map(line_count_src, src_lines, chunksize=64)
                pool.close()
                pool.join()

            tgt_lines = list(tgt.readlines())
            print(f"{split}.tgt data_num: {len(tgt_lines)}")
            with Pool(processes=8) as pool:
                results_tgt = pool.map(line_count_tgt, tgt_lines)
                pool.close()
                pool.join()

            assert len(results_tgt) == len(results_src)
            data_num = len(results_tgt)
            total_sent_num_src = 0
            total_token_num_src = 0
            total_sent_num_tgt = 0
            total_token_num_tgt = 0
            for i in range(data_num):
                result_src = results_src[i]
                result_tgt = results_tgt[i]
                total_sent_num_src += result_src[0]
                total_token_num_src += result_src[1]
                total_sent_num_tgt += result_tgt[0]
                total_token_num_tgt += result_tgt[1]

            result = {
                "avg_sent_num_src": total_sent_num_src / data_num,
                "avg_token_num_src": total_token_num_src / data_num,
                "avg_sent_num_tgt": total_sent_num_tgt / data_num,
                "avg_token_num_tgt": total_token_num_tgt / data_num,
            }
            results[split] = result

    with open(os.path.join(data_dir, 'length.json'), 'w') as f:
        f.write(json.dumps(results, indent=4))
        print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/data1/tsq/contrastive/clustering_3/raw_dataset/animal')
    args = parser.parse_args()
    check_align(args)
