# coding: utf-8

import argparse
import os
import json
import logging

logging.basicConfig(level=logging.INFO)

from multiprocessing import Pool
from src.group.data import load_data_from_file


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


def count_length_wikicatsum(args):
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
                "data_num": data_num
            }
            results[split] = result
    total_res = {
        "avg_sent_num_src": 0,
        "avg_token_num_src": 0,
        "avg_sent_num_tgt": 0,
        "avg_token_num_tgt": 0,
        "data_num": 0
    }
    for split, result in results.items():
        data_num = result['data_num']
        for k, v in result.items():
            if k == 'data_num':
                total_res[k] += v
            else:
                total_res[k] += v * data_num
    total_data_num = total_res['data_num']
    for k, v in total_res.items():
        if k != 'data_num':
            total_res[k] = v / total_data_num

    with open(os.path.join(data_dir, 'length.json'), 'w') as f:
        f.write(json.dumps(results, indent=4))
        print(json.dumps(results, indent=4))

    with open(os.path.join(data_dir, 'total_length.json'), 'w') as f:
        f.write(json.dumps(total_res, indent=4))
        print(json.dumps(total_res, indent=4))


def count_length_multi_news(args):
    loaders = load_data_from_file(
        args.data_dir, batch_size=1, rouge_path='', logger=None,
        max_sum_sent_num=100000000000)
    splits = ['train', 'valid', 'test']
    results = {}
    for i, split in enumerate(splits):
        loader = loaders[i]
        data_num = len(loader)
        total_sent_num_src = 0
        total_token_num_src = 0
        total_longest_sent_len_src = 0

        total_sent_num_tgt = 0
        total_token_num_tgt = 0
        total_longest_sent_len_tgt = 0

        for batch_iter, batch in enumerate(loader):
            input_sent_embeddings, doc, summ, doc_len, summ_len, rouge_scores = [
                a for a in batch[0]]
            total_sent_num_src += doc.shape[0]
            total_sent_num_tgt += summ.shape[0]
            total_token_num_src += doc_len.numpy().sum()
            total_token_num_tgt += summ_len.numpy().sum()
            total_longest_sent_len_src += doc.shape[1]
            total_longest_sent_len_tgt += summ.shape[1]

        result = {
            "avg_sent_num_src": total_sent_num_src / data_num,
            "avg_token_num_src": total_token_num_src / data_num,
            "avg_longest_sent_len_src": total_longest_sent_len_src / data_num,
            "avg_sent_num_tgt": total_sent_num_tgt / data_num,
            "avg_token_num_tgt": total_token_num_tgt / data_num,
            "avg_longest_sent_len_tgt": total_longest_sent_len_tgt / data_num,
        }
        results[split] = result

    # with open(os.path.join(data_dir, 'length.json'), 'w') as f:
    #     f.write(json.dumps(results, indent=4))
    print(json.dumps(results, indent=4))


def count_length_wcep(args):
    data_dir = os.path.join(args.data_dir)
    splits = ['train', 'val', 'test']
    total_sent_num_src_all = 0
    total_data_num = 0
    for split in splits:
        total_sent_num_src = 0
        with open(os.path.join(data_dir, f"{split}_titles.txt")) as src:
            data_num = len(src.readlines())
            src_dir = os.path.join(data_dir, f"{split}_src")
            for i in range(data_num):
                with open(os.path.join(src_dir, f"{i}.txt"), 'r') as fin:
                    total_sent_num_src += len(fin.readlines())
        total_sent_num_src_all += total_sent_num_src
        total_data_num += data_num
        print(split)
        print("total_sent_num_src is")
        print(total_sent_num_src / data_num)
    print("all")
    print(total_sent_num_src_all)
    print(total_data_num)
    print(total_sent_num_src_all / total_data_num)


if __name__ == "__main__":
    datasets = {
        'wikicatsum': count_length_wikicatsum,
        'multi_news': count_length_multi_news,
        'wcep': count_length_wcep
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='wikicatsum', choices=datasets.keys())
    parser.add_argument('--data-dir', type=str,
                        default='/data/tsq/contrastive/clust_documents/wcep/raw'
                        # default='/data1/tsq/contrastive/clustering_3/raw_dataset/film'
                        # default='/data1/tsq/contrastive/group/multi_news/text.pkl'
                        )
    args = parser.parse_args()
    datasets[args.data](args)
