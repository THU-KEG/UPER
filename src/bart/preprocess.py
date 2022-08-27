# coding: utf-8
import json
from multiprocessing import Pool
from tqdm import tqdm
from transformers import BartTokenizer
import argparse
import os
import torch
import logging
from src.label.preprocess import check_fout, mkdir
from src.SimCLS.multi_doc_prepare import tgt_copy_after_ignore

logging.basicConfig(level=logging.INFO)
splits = ['train', 'valid', 'test']
sp2id = {'train': 0, 'valid': 1, 'test': 2}


def merge_rerank_documents(diverse_dir, model_pt, bart_data_dir):
    for split in splits:
        doc_fout = check_fout(os.path.join(bart_data_dir, f"{split}.source"))
        ref_fout = check_fout(os.path.join(bart_data_dir, f"{split}.target"))
        # read from rerank result
        result_dir = os.path.join(diverse_dir, f"result_{split}", model_pt)
        candidate_dir = os.path.join(result_dir, "candidate")
        reference_dir = os.path.join(result_dir, "reference")
        data_num = len(os.listdir(reference_dir))
        assert len(os.listdir(candidate_dir)) == data_num
        for data_id in range(data_num):
            candidate_fin = open(os.path.join(candidate_dir, f"{data_id}.dec"), "r")
            reference_fin = open(os.path.join(reference_dir, f"{data_id}.ref"), "r")
            # doc
            doc_lines = candidate_fin.readlines()
            cleaned_doc_lines = [line.strip() for line in doc_lines]
            doc_fout.write(" ".join(cleaned_doc_lines))
            doc_fout.write("\n")

            # ref
            ref_lines = reference_fin.readlines()
            cleaned_ref_lines = [line.strip() for line in ref_lines]
            ref_fout.write(" ".join(cleaned_ref_lines))
            ref_fout.write("\n")


def create_bart_documents(wikicatsum_dir, category, bart_data_dir):
    for split in splits:
        # get ignore list
        origin_dir = os.path.join(wikicatsum_dir, category)
        if split == 'train':
            ignore_filename = os.path.join(origin_dir, 'ignoredIndices.log')
        else:
            ignore_filename = os.path.join(origin_dir, '%s_ignoredIndices.log' % split)

        ignore_list = set()
        if os.path.exists(ignore_filename):
            with open(ignore_filename, 'r') as fin:
                for line in fin:
                    ignore_list.add(int(line.strip()))
        logging.info(f"ignore_list has:{len(ignore_list)}")
        # file path
        new_src_fout = check_fout(os.path.join(bart_data_dir, f"{split}.source"))
        # for source documents
        src_path = os.path.join(origin_dir, f"{split}.src")
        with open(src_path, 'r') as fin:
            lines = fin.readlines()
            for i, line in tqdm(enumerate(lines)):
                if i in ignore_list:
                    continue
                else:
                    title_and_documents = line.split("<EOT>")
                    documents_str = title_and_documents[1]
                    documents = documents_str.split("<EOP>")
                    cleaned_docs = [document.strip() for document in documents]
                    new_src_fout.write(" ".join(cleaned_docs))
                    new_src_fout.write('\n')

        tgt_copy_after_ignore(origin_dir, f"{split}.tgt", bart_data_dir, f"{split}.target", ignore_list, tokenized=True)


def work(args, save_dir):
    try:
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_dir)
    except ValueError:
        tokenizer = BartTokenizer.from_pretrained('/data/tsq/contrastive/pretrained_models/bart_base')

    data = [[], [], []]

    if args.split_mode == '3split':
        splits = ['train', 'valid', 'test']
    elif args.split_mode == 'test_as_valid':
        splits = ['train', 'test']
    else:
        # only test
        splits = ['test']

    for i, sp in enumerate(splits):
        if args.category == "multi_news":
            data_file_src = os.path.join(save_dir, '%s.txt.src.tokenized.fixed.cleaned.final.truncated.txt' % (sp))
            data_file_tgt = os.path.join(save_dir, '%s.txt.tgt.tokenized.fixed.cleaned.final.truncated.txt' % (sp))
        else:
            data_file_src = os.path.join(save_dir, '%s.source' % (sp))
            data_file_tgt = os.path.join(save_dir, '%s.target' % (sp))
        empty_seg_num = 0
        with open(data_file_src, 'r', encoding='utf-8') as fin_src:
            lines_src = fin_src.readlines()
            fin_tgt = open(data_file_tgt, 'r', encoding='utf-8')
            lines_tgt = fin_tgt.readlines()
            assert len(lines_src) == len(lines_tgt)

            total_len = len(lines_tgt)
            for j in tqdm(range(total_len), desc=f"{sp}"):
                try:
                    doc_seg = lines_src[j].strip()
                    abs_seg = lines_tgt[j].strip()


                except:
                    print("###################")
                    print(doc_seg)
                    print("error")
                    print(abs_seg)
                    continue
                    quit()
                # check if there is empty doc/abs
                if len(doc_seg) == 0 or len(abs_seg) == 0:
                    data[sp2id[sp]].append((None, None, None))
                    empty_seg_num += 1
                    continue

                max_input_length = args.max_len
                # add [] on doc_seg, because we want to create a batch whose size is 1
                tokenize_res = tokenizer.batch_encode_plus([doc_seg], max_length=max_input_length, return_tensors="pt",
                                                           pad_to_max_length=True, truncation=True)
                max_target_length = args.max_tgt_len
                with tokenizer.as_target_tokenizer():
                    if sp == 'train':
                        # when training, we will truncate the target
                        tgt_tokenize_res = tokenizer.batch_encode_plus([abs_seg], max_length=max_target_length,
                                                                       return_tensors="pt",
                                                                       pad_to_max_length=False, truncation=True)
                    else:
                        tgt_tokenize_res = tokenizer.batch_encode_plus([abs_seg], return_tensors="pt")
                tokenize_res["labels"] = tgt_tokenize_res['input_ids']

                data[sp2id[sp]].append(
                    (tokenize_res['input_ids'], tokenize_res['attention_mask'], tokenize_res["labels"]))

        logging.info(f"There are {empty_seg_num} empty segments in {data_file_src} and {data_file_tgt}")
    if args.split_mode == 'test_as_valid':
        # [Note] this is a shallow copy. if change data[2], data[1] will change, too
        data[1] = data[2]
    torch.save(data, os.path.join(save_dir, 'bart_data_ml%d_mtl%d.pt' % (args.max_len, args.max_tgt_len)))
    logging.info(f"Finish preprocess for bart on {save_dir} by {args.split_mode}")


def create_wcep_documents(root_dir, truncate_strategy, save_dir):
    raw_data_dir = os.path.join(root_dir, 'raw')
    splits = ['train', 'val', 'test']

    for i, sp in enumerate(splits):
        # file path
        src_file = os.path.join(save_dir, f"{sp}.source")
        tgt_file = os.path.join(save_dir, f"{sp}.target")
        new_src_fout = check_fout(src_file)
        new_tgt_fout = check_fout(tgt_file)
        input_json_path = os.path.join(raw_data_dir, '{}.jsonl'.format(sp))
        lines = open(input_json_path, 'r').readlines()
        for line in lines:
            json_dict = json.loads(line)
            # tgt
            summary = json_dict['summary']
            new_tgt_fout.write(summary)
            new_tgt_fout.write("\n")

            # src
            src_texts = []
            for article in json_dict["articles"]:
                text = article['text']
                src_texts.append(text.replace('\n', ' ').strip())
            new_src_fout.write(' '.join(src_texts))
            new_src_fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bart model preprocess.')
    # Data parameters
    parser.add_argument('--max-tgt-len', type=int, default=256)
    parser.add_argument('--max-len', type=int, default=1000)
    parser.add_argument('--num_topics', type=int, default=3)
    parser.add_argument('--single_topic_id', type=int, default=-1, help="if > -1, we will process that topic only")

    # S/L parameters
    parser.add_argument('--tokenizer-dir', type=str, default='facebook/bart-base')
    parser.add_argument('--wikicatsum-dir', type=str, default='/data/tsq/wikicatsum')
    parser.add_argument('--topicdata-dir', type=str, default=None)
    parser.add_argument("--source", choices=['no_rank', "doc_rerank"], help="use no rerank strategy")
    # If use rerank documents, these parameters are needed:
    parser.add_argument("--model_pt", default="21-07-02-3", type=str)
    parser.add_argument('--output_dir', type=str, default='/data/tsq/contrastive/clust_documents/')
    parser.add_argument('--category', type=str, choices=['animal', 'company', 'film', 'multi_news', 'wcep', 'copy'],
                        default='wcep')
    parser.add_argument('--split_mode', type=str, default="3split",
                        choices=['3split', 'test_as_valid', 'test_only'],
                        help='how to use data split')
    # for wcep
    parser.add_argument("--truncate_strategy", default="origin", choices=['origin', 'first_sent'],
                        help="truncate strategy")

    args = parser.parse_args()

    if args.source == 'no_rank':
        if args.category == "multi_news" or args.category == "copy":
            args.bart_data_dir = args.output_dir
        elif args.category == "wcep":
            wcep_root = os.path.join(args.output_dir, 'wcep')
            args.bart_data_dir = os.path.join(wcep_root, 'raw', args.truncate_strategy)
            mkdir(args.bart_data_dir)
            create_wcep_documents(wcep_root, args.truncate_strategy, args.bart_data_dir)

        else:
            args.bart_data_dir = os.path.join(args.output_dir, args.category, 'bart_no_rerank')
            mkdir(args.bart_data_dir)
            create_bart_documents(args.wikicatsum_dir, args.category, args.bart_data_dir)
        work(args, args.bart_data_dir)
    elif args.source == "prompt":
        pass

    elif args.source == "doc_rerank":
        # rerank documents
        args.bart_data_dir = os.path.join(args.output_dir, args.category, 'diverse', f'bart_{args.model_pt}')
        mkdir(args.bart_data_dir)
        args.diverse_dir = os.path.join(args.output_dir, args.category, 'diverse')
        merge_rerank_documents(args.diverse_dir, args.model_pt, args.bart_data_dir)
        work(args, args.bart_data_dir)
    else:
        # rerank the candidate summary
        assert args.num_topics == len(os.listdir(args.topicdata_dir))

        lst = []
        for topic_id in range(args.num_topics):
            save_dir = os.path.join(args.topicdata_dir, f'topic{topic_id}')
            lst.append((args, save_dir))

        if args.single_topic_id > -1:
            save_dir = os.path.join(args.topicdata_dir, f'topic{args.single_topic_id}')
            work(args, save_dir)
        else:
            # multiprocess
            with Pool(processes=args.num_topics) as pool:
                rslt = pool.starmap(work, lst)
