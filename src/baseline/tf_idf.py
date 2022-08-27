import argparse
import os
import shutil
import copy
import torch
from tqdm import tqdm
from src.statistics.reproduce import call_rouge
from src.clust.test import copy_ref, make_dirs
from src.bart.preprocess import work as preprocess
from src.bart.preprocess import check_fout


def extract_final_summary(trunc_result_dir, split, tgt_dir):
    # path
    sum_dir = os.path.join(trunc_result_dir, "sum")
    ref_dir = os.path.join(trunc_result_dir, "ref")
    data_num = len(os.listdir(sum_dir))
    assert len(os.listdir(ref_dir)) == data_num

    # output
    tgt_ext_path = os.path.join(tgt_dir, f"{split}.source")
    tgt_ref_path = os.path.join(tgt_dir, f"{split}.target")
    ext_fout = check_fout(tgt_ext_path)
    ref_fout = check_fout(tgt_ref_path)

    for data_id in tqdm(range(data_num)):
        # read sorted sentences
        sum_path = os.path.join(sum_dir, f'{data_id}_decoded.txt')
        with open(sum_path, 'r') as fin_sum:
            sum_line = fin_sum.read().strip()
            ext_fout.write(sum_line)
            ext_fout.write("\n")
        # read ref
        ref_path = os.path.join(ref_dir, f'{data_id}_reference.txt')
        with open(ref_path, 'r') as fin_ref:
            ref_lines = fin_ref.readlines()
            final_ref_lines = [line.strip() for line in ref_lines]
            ref_fout.write(" ".join(final_ref_lines))
            ref_fout.write("\n")

    print(f"[1]extracted sentences of {split} set are read from {trunc_result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bart model preprocess.')
    # Data parameters
    parser.add_argument('--max_token_num', type=int, default=100)
    # S/L parameters
    parser.add_argument('--task', type=str, default="ext_abs",
                        choices=['ext', 'ext_abs', 'tokenize'],
                        help='what is task')

    parser.add_argument('--split_mode', type=str, default="3split",
                        choices=['3split', 'test_as_valid', 'test_only'],
                        help='how to use data split')

    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--bart_data_dir', type=str, default="/data1/tsq/contrastive/rerank_documents",
                        help='dir of raw data ')
    parser.add_argument('--data_dir', type=str, default="/data1/tsq/contrastive/clust_documents",
                        help='dir of output data')

    # Data parameters for bart
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--tokenizer-dir', type=str, default='facebook/bart-base')
    parser.add_argument('--max-tgt-len', type=int, default=256, help="when training, we will truncate the target")

    args = parser.parse_args()
    print(args)

    if args.task == 'ext':
        src_path = os.path.join(args.bart_data_dir, args.category, 'bart_no_rerank', 'test.source')
        tgt_dir = os.path.join(args.data_dir, args.category, 'result', 'tf_idf', f'mtn{args.max_token_num}')
        fin = open(src_path, 'r')
        src_lines = fin.readlines()
        new_src_dir = os.path.join(tgt_dir, 'sum')
        make_dirs(new_src_dir)
        for i, line in enumerate(src_lines):
            new_src_path = os.path.join(new_src_dir, f"{i}_decoded.txt")
            tokens = line.split()[:args.max_token_num]
            new_line = " ".join(tokens)
            with open(new_src_path, "w") as fout:
                fout.write(new_line)

        copy_ref(tgt_dir, 'test', 0, len(src_lines), args.category)
        call_rouge(tgt_dir)
    else:
        # ext-abs
        if args.split_mode == '3split':
            splits = ['train', 'valid', 'test']
        elif args.split_mode == 'test_as_valid':
            splits = ['train', 'test']
        else:
            # only test
            splits = ['test']
        src_dir = os.path.join(args.data_dir, args.category, 'result', 'tf_idf', f'mtn{args.max_token_num}')
        tgt_dir = os.path.join(args.data_dir, args.category, 'bart', args.split_mode, 'tf_idf')
        if not os.path.exists(tgt_dir):
            make_dirs(tgt_dir)

        if args.task != 'tokenize':
            for split in splits:
                origin_src_path = os.path.join(args.bart_data_dir, args.category, 'bart_no_rerank',
                                               f'{split}.source')
                origin_tgt_path = os.path.join(args.bart_data_dir, args.category, 'bart_no_rerank',
                                               f'{split}.target')
                tgt_ext_path = os.path.join(tgt_dir, f"{split}.source")
                tgt_ref_path = os.path.join(tgt_dir, f"{split}.target")
                shutil.copyfile(origin_src_path, tgt_ext_path)
                shutil.copyfile(origin_tgt_path, tgt_ref_path)
        # tokenize them by bart
        bart_args = copy.deepcopy(args)
        preprocess(bart_args, tgt_dir)
        print("[2]Finish bart preprocess at ", tgt_dir)
