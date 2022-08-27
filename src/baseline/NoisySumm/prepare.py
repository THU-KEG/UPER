import argparse
import os
import shutil
import copy
import json
from tqdm import tqdm
from src.statistics.reproduce import call_rouge
from src.clust.test import copy_ref, make_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unilm model preprocess.')
    # S/L parameters
    parser.add_argument('--extractor', type=str, default="tf_idf",
                        choices=['tf_idf', 'power'],
                        help='what is extractor')
    parser.add_argument('--extractor_dir', type=str, required=True,
                        help='dir of extractor result')

    # /data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/inverse_add4/ws_0.75/inverse_add_title

    parser.add_argument('--split_mode', type=str, default="test_as_valid",
                        choices=['3split', 'test_as_valid', 'test_only'],
                        help='how to use data split')

    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--data_dir', type=str, default="/data1/tsq/contrastive/clust_documents",
                        help='dir of output data')
    parser.add_argument('--result_dir', type=str, default="/data1/tsq/unilm/input",
                        help='dir of result')
    args = parser.parse_args()
    print(args)

    # ext-abs
    if args.split_mode == '3split':
        splits = ['train', 'valid', 'test']
    elif args.split_mode == 'test_as_valid':
        splits = ['train', 'test']
    else:
        # only test
        splits = ['test']

    tgt_dir = os.path.join(args.result_dir, args.category, args.extractor)
    if not os.path.exists(tgt_dir):
        make_dirs(tgt_dir)

    for split in splits:
        origin_src_path = os.path.join(args.extractor_dir, f'{split}.source')
        origin_tgt_path = os.path.join(args.extractor_dir, f'{split}.target')
        tgt_ext_path = os.path.join(tgt_dir, f"{split}.json")
        with open(origin_src_path, 'r') as src_fin:
            with open(origin_tgt_path, 'r') as tgt_fin:
                with open(tgt_ext_path, 'a') as fout:
                    src_lines = src_fin.readlines()
                    tgt_lines = tgt_fin.readlines()
                    data_num = len(src_lines)
                    assert len(tgt_lines) == data_num
                    for data_id in range(data_num):
                        src_line = src_lines[data_id].strip()
                        tgt_line = tgt_lines[data_id].strip()
                        data_dict = {"src": src_line, "tgt": tgt_line}
                        fout.write(json.dumps(data_dict))
                        fout.write('\n')
