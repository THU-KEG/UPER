import torch
import argparse
import os
import json
import shutil
from difflib import SequenceMatcher
from tqdm import tqdm
from src.bart.pyrouge.rouge import Rouge155
from src.bart.model import BartDecodeModel
from src.bart.data import TopicDataset
from src.bart.utils import setup_logger, MetricLogger, strip_prefix_if_present


def validate(data, model, num_beams, max_length, min_length, device, log_dir, fast=True):
    model.eval()
    torch.cuda.empty_cache()

    ref_dir = os.path.join(log_dir, 'ref')
    sum_dir = os.path.join(log_dir, 'sum')
    candidate_dir = os.path.join(log_dir, 'candidate')
    src_dir = os.path.join(log_dir, 'src')

    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    os.mkdir(ref_dir)
    if os.path.isdir(sum_dir):
        shutil.rmtree(sum_dir)
    os.mkdir(sum_dir)
    if os.path.isdir(candidate_dir):
        shutil.rmtree(candidate_dir)
    os.mkdir(candidate_dir)
    if os.path.isdir(src_dir):
        shutil.rmtree(src_dir)
    os.mkdir(src_dir)

    tokenizer = model.bart_tokenizer
    with torch.no_grad():
        try:
            for batch_iter, batch in tqdm(enumerate(data.gen_batch()), desc="Validating", total=data.example_num):
                if fast and batch_iter % 10 != 0:
                    continue
                if batch[0] == None:
                    # empty input
                    continue

                input_ids, attention_mask, labels = [a.to(device) for a in batch]
                goldens = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                           labels]
                golden = goldens[0]
                pred_candidate_list = model(input_ids, attention_mask, None, num_beams, max_length, min_length, device)

                # pred = strip_redundant_sentences(pred_candidate_list[0])
                pred = pred_candidate_list[0]

                with open(os.path.join(ref_dir, "%d_reference.txt" % (batch_iter)), 'w') as f:
                    f.write(golden)
                with open(os.path.join(sum_dir, "%d_decoded.txt" % (batch_iter)), 'w') as f:
                    f.write(pred)
                if not fast:
                    # output documents and candidate summaries of each topic
                    with open(os.path.join(candidate_dir, "%d_candidates.txt" % (batch_iter)), 'w') as f:
                        candidates = "\n".join(pred_candidate_list)
                        f.write(candidates)
                    # out src
                    srcs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            input_ids]
                    src = srcs[0]
                    with open(os.path.join(src_dir, "%d_src.txt" % (batch_iter)), 'w') as f:
                        f.write(src)
        except RuntimeError:
            # In python 3.7, if batch iterator raise StopIteration, it will be like a RuntimeError
            print(f"stop at {batch_iter} when validating")

    Rouge155_obj = Rouge155(stem=True, tmp=os.path.join(log_dir, 'tmp'))
    score = Rouge155_obj.evaluate_folder(sum_dir, ref_dir)
    return score


def vis_scores(scores):
    recall_keys = {'rouge_1_recall', 'rouge_2_recall', 'rouge_l_recall', 'rouge_su4_recall'}
    f_keys = {'rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score'}
    if type(list(scores.values())[0]) == dict:
        for n in scores:
            if n == 'all':
                scores[n] = {k: scores[n][k] for k in f_keys}
            else:
                scores[n] = {k: scores[n][k] for k in recall_keys}
    else:
        scores = {k: scores[k] for k in f_keys}
    return json.dumps(scores, indent=4)


def call_validate(args):
    device = 'cuda'

    if (args.test):
        data = TopicDataset('test', args.data_path, max_len=args.max_len, max_target_len=args.max_tgt_len)
    else:
        data = TopicDataset('valid', args.data_path, max_len=args.max_len, max_target_len=args.max_tgt_len)
    data_base = os.path.splitext(os.path.basename(args.data_path))[0]

    model_kwargs = {"model_name": args.model_name}
    if args.raw:
        model = BartDecodeModel(**model_kwargs)
        if args.test:
            log_dir = os.path.join(args.data_path, 'test_%s_%s' % (data_base, args.model_name.split('/')[-1]))
        else:
            log_dir = os.path.join(args.data_path, 'val_%s_%s' % (data_base, args.model_name.split('/')[-1]))
    else:
        loaded = torch.load(args.ckpt)
        # for k in model_kwargs:
        #     if hasattr(args, k) and getattr(args, k) is not None:
        #         model_kwargs[k] = getattr(args, k)
        loaded['state_dict'] = strip_prefix_if_present(loaded['state_dict'], prefix='module.')
        for k, v in model_kwargs.items():
            print(k, v)

        model = BartDecodeModel(**model_kwargs)
        model.load_state_dict(loaded['state_dict'])
        ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]

        if args.test:
            log_dir = os.path.join(os.path.dirname(args.ckpt), 'test_%s_%s' % (data_base, ckpt_base))
        else:
            log_dir = os.path.join(os.path.dirname(args.ckpt), 'val_%s_%s' % (data_base, ckpt_base))

    model = model.to(device)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    else:
        print('log dir %s exists, be careful that we will overwrite it' % log_dir)

    scores = validate(data, model, args.beam_size, args.max_dec_len, args.min_dec_len, device, log_dir, fast=args.fast)
    with open(os.path.join(log_dir, 'scores.txt'), 'w') as f:
        for k, v in model_kwargs.items():
            f.write('%s: %s\n' % (k, str(v)))
        f.write(json.dumps(scores, indent=4))
    vis = vis_scores(scores)
    print(vis)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt', required=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    parser.add_argument('--ckpt', default="/data1/tsq/WikiGen/bart_base/animal/model_epoch2_val0.388.pt")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--fast', action="store_true")
    parser.add_argument('--raw', action="store_true", help="whether use the un-fine-tuned model")
    # parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--data_path', default='/data1/tsq/wikicatsum/animal/bart_base_data',
                        help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--max-tgt-len', type=int, default=256)

    # decode parameters
    parser.add_argument('--model-name', type=str, default='facebook/bart-base')
    parser.add_argument('--min_dec_len', type=int, default=55)
    parser.add_argument('--max_dec_len', type=int, default=120)
    parser.add_argument('--max_dec_sent', type=int, default=15)
    parser.add_argument('--beam_size', type=int, default=16)
    args = parser.parse_args()

    # args.data_path = '/data1/tsq/wikicatsum/%s/bart_data' % args.category
    # args.data_path = '/data1/tsq/wikicatsum/%s/bart_base_data' % args.category
    call_validate(args)


if __name__ == "__main__":
    main()
