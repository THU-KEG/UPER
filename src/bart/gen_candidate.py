# coding: utf-8
from multiprocessing import Pool
from tqdm import tqdm
from transformers import BartTokenizer
import shutil
import argparse
import os
import torch
from src.bart.data import TopicDataset
from src.bart.model import BartDecodeModel
from src.bart.utils import strip_prefix_if_present
from src.label.preprocess import check_fout, mkdir
import logging

logging.basicConfig(level=logging.INFO)


def work(args, save_dir, cuda_id):
    if args.single_topic_id == -1:
        # multiprocess
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{cuda_id}'
        # candidate length for each topic
        args.max_candidate_len = args.max_candidate_len // args.num_topics
        args.min_candidate_len = args.min_candidate_len // args.num_topics

    device = args.device
    splits = ['test', 'valid', 'train']

    if args.raw_model:
        model = BartDecodeModel(args.model_name)
    else:
        ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
        ckpt_output_dir = os.path.join(os.path.dirname(args.ckpt), 'candidate_beam%d_%s' % (args.num_beams, ckpt_base))

        loaded = torch.load(args.ckpt)
        # model_kwargs = loaded['kwargs']
        model_kwargs = {"model_name": args.model_name}
        # for k in model_kwargs:
        #     if hasattr(args, k) and getattr(args, k) is not None:
        #         model_kwargs[k] = getattr(args, k)
        loaded['state_dict'] = strip_prefix_if_present(loaded['state_dict'], prefix='module.')
        for k, v in model_kwargs.items():
            print(k, v)

        model = BartDecodeModel(**model_kwargs)
        model.load_state_dict(loaded['state_dict'])

    model = model.to(device)
    model.eval()

    for i, sp in enumerate(splits):
        data_loader = TopicDataset(sp, save_dir, shuffle=False, max_len=args.max_len)
        tokenized_cand_out_path = os.path.join(save_dir, f"{sp}.out.tokenized")
        cand_out_path = os.path.join(save_dir, f"{sp}.out")
        tokenized_cand_fout = check_fout(tokenized_cand_out_path)
        cand_fout = check_fout(cand_out_path)
        try:
            with torch.no_grad():
                for batch_iter, batch in tqdm(enumerate(data_loader.gen_batch()), desc=f"generating {sp} candidates ",
                                              total=data_loader.example_num):

                    if batch[0] == None:
                        # empty input
                        pred_candidate_list = ['' for i in range(args.num_beams)]
                    else:
                        input_ids, attention_mask, labels = [a.to(device) for a in batch]
                        pred_candidate_list = model(input_ids, attention_mask, labels=None, num_beams=args.num_beams,
                                                    max_length=args.max_candidate_len,
                                                    min_length=args.min_candidate_len)

                    # TODO strip redundant sentences or not ?
                    # pred = strip_redundant_sentences(pred)

                    # output
                    output_txt(cand_fout, pred_candidate_list)
                    if batch[0] == None:
                        # no need to tokenize empty string
                        tokenized_candidate_list = pred_candidate_list
                    else:
                        tokenized_candidate_list = tokenize_candidates(pred_candidate_list)
                    output_txt(tokenized_cand_fout, tokenized_candidate_list)
        except RuntimeError:
            # In python 3.7, if batch iterator raise StopIteration, it will be like a RuntimeError
            logging.info(f"stop at {batch_iter} on {sp} set")
            if not args.raw_model:
                _copy(ckpt_output_dir, tokenized_cand_out_path, cand_out_path, sp)

    logging.info(f"Finish candidate generation on {save_dir}")


def _copy(ckpt_output_dir, tokenized_cand_out_path, cand_out_path, sp):
    mkdir(ckpt_output_dir)
    new_tokenized_cand_out_path = os.path.join(ckpt_output_dir, f"{sp}.out.tokenized")
    new_cand_out_path = os.path.join(ckpt_output_dir, f"{sp}.out")
    shutil.copyfile(tokenized_cand_out_path, new_tokenized_cand_out_path)
    shutil.copyfile(cand_out_path, new_cand_out_path)


def output_txt(fout, candidate_list):
    for line in candidate_list:
        fout.write(line)
        fout.write('\n')


def tokenize_candidates(pred_candidate_list):
    tokenized_candidate_list = []
    for line in pred_candidate_list:
        tokens = line.strip().split()
        new_tokens = [token.lower() for token in tokens]
        new_line = ' '.join(new_tokens)
        tokenized_candidate_list.append(new_line)
    return tokenized_candidate_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bart model preprocess.')
    # Data parameters
    parser.add_argument('--max-tgt-len', type=int, default=256)
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--num_topics', type=int, default=3)
    parser.add_argument('--num_beams', type=int, default=16)
    parser.add_argument('--max-candidate-len', type=int, default=112,
                        help="if multiprocess, we will divide it by num_topics")
    parser.add_argument('--min-candidate-len', type=int, default=55,
                        help="if multiprocess, we will divide it by num_topics")

    # S/L parameters
    parser.add_argument('--topicdata-dir', type=str,
                        default='/data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3')
    # parser.add_argument('--model-name', type=str, default='facebook/bart-large')
    parser.add_argument('--model-name', type=str, default='facebook/bart-base')
    parser.add_argument('--single_topic_id', type=int, default=-1, help="if > -1, we will process that topic only")

    # ways of generation
    # parser.add_argument('--ckpt', default="/data1/tsq/WikiGen/bart_base/animal/model_epoch4_val0.392.pt")
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--raw_model', action="store_true", help="if true, we will use bart which wasn't fine-tuned")

    # cuda
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--available_cuda', type=int, nargs='+', default=[3, 4, 5, 6])

    args = parser.parse_args()

    assert args.num_topics == len(os.listdir(args.topicdata_dir))

    lst = []
    for topic_id in range(args.num_topics):
        save_dir = os.path.join(args.topicdata_dir, f'topic{topic_id}')
        cuda_id = args.available_cuda[topic_id]
        lst.append((args, save_dir, cuda_id))

    if args.single_topic_id > -1:
        save_dir = os.path.join(args.topicdata_dir, f'topic{args.single_topic_id}')
        work(args, save_dir, 0)
    else:
        # multiprocess
        with Pool(processes=args.num_topics) as pool:
            rslt = pool.starmap(work, lst)
