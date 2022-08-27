import argparse
from tqdm import tqdm
import os
import string
import json
import shutil
from src.bart.pyrouge.rouge import Rouge155
import nltk
from src.bart.validate import vis_scores

_tok_dict = {}


def _is_digit(w):
    for ch in w:
        if not (ch.isdigit() or ch == ','):
            return False
    return True


def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and \
                input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'" + input_tokens[i + 1])
            i += 2
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(
                input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ',' + input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and \
                input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.' + input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[
            -1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[
            i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i + 3
            while k + 2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)


def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter


def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)


def remove_duplicate(l_list, duplicate_rate):
    tk_list = [l.lower().split() for l in l_list]
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set & history_set) / len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list


def process_eval(args, log_dir):
    result_dir = os.path.join(args.save_dir, 'test_result')
    _strategy_name = "_".join([strate[:3] for strate in args.strategy])
    trick_result_dir = os.path.join(result_dir, f"mbn{args.max_beam_num}_dr{args.duplicate_rate}_{_strategy_name}")
    if not os.path.exists(trick_result_dir):
        os.makedirs(trick_result_dir)

    # read gold
    if args.model_name == 'unilm':
        wikicatsum_dir = os.path.join(args.wikicatsum_dir, args.category)
        gold_path = os.path.join(wikicatsum_dir, 'test.tgt')
        ignore_path = os.path.join(wikicatsum_dir, 'test_ignoredIndices.log')
        gold_list = []
        ignore_lines = open(ignore_path, 'r').readlines()
        ignore_idxs = [int(ignore_line.strip()) for ignore_line in ignore_lines]
        with open(gold_path, 'r') as fin:
            tgt_lines = fin.readlines()
            for idx, tgt_line in enumerate(tgt_lines):
                if idx in ignore_idxs:
                    continue
                if 'split_ref' in args.strategy:
                    tgt_line = tgt_line.strip().replace("<SNT>", '\n')
                gold_list.append(tgt_line)
        data_num = len(gold_list)
        src_path = os.path.join(args.data_path, '.test')
        src_lines = open(src_path, 'r').readlines()
        assert len(src_lines) == data_num
    else:
        gold_dir = os.path.join(log_dir, 'ref')
        candidate_dir = os.path.join(log_dir, 'candidate')
        data_num = len(os.listdir(gold_dir))
        assert data_num == len(os.listdir(candidate_dir))
        gold_list = []
        for i in range(data_num):
            old_ref_path = os.path.join(gold_dir, f"{i}_reference.txt")
            with open(old_ref_path, "r") as f_in:
                line = f_in.read()
                if 'split_ref' in args.strategy:
                    line = line.strip().replace(" . ", ' .\n')
                gold_list.append(line)

    # process predict candidates
    pred_list = []
    for data_id in range(data_num):
        if args.model_name == 'unilm':
            candidate_str = src_lines[data_id]
        else:
            old_candidate_path = os.path.join(candidate_dir, f"{data_id}_candidates.txt")
            with open(old_candidate_path, "r") as f_in:
                # there are beam_size candidates, take top max_beam_num ones
                candidates = f_in.readlines()[:args.max_beam_num]
                candidate_str = ' '.join(candidates)
        buf = []
        for sentence in candidate_str.strip().split("."):
            if 'fix_tokenization' in args.strategy:
                sentence = fix_tokenization(sentence)
            while "  " in sentence:
                sentence = sentence.replace("  ", " ")
            if any(get_f1(sentence, s) > 1.0 for s in buf):
                continue
            s_len = len(sentence.split())
            if s_len <= 4:
                continue
            buf.append(sentence.strip() + " .")

        if args.duplicate_rate and args.duplicate_rate < 1 and 'remove_redundant' in args.strategy:
            buf = remove_duplicate(buf, args.duplicate_rate)
        trunc_list = buf
        print(f"{data_id} final result sent num is {len(trunc_list)}")

        line = "\n".join(trunc_list)

        print(f"{data_id} final result token num is {len(line.strip().split())}")
        pred_list.append(line)

    # output and call rouge
    output(trick_result_dir, gold_list, pred_list, args)


def output(trick_result_dir, gold_list, pred_list, args):
    ref_dir = os.path.join(trick_result_dir, 'ref')
    sum_dir = os.path.join(trick_result_dir, 'sum')
    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    os.mkdir(ref_dir)
    if os.path.isdir(sum_dir):
        shutil.rmtree(sum_dir)
    os.mkdir(sum_dir)
    data_num = len(pred_list)
    assert len(gold_list) == data_num
    for data_id in range(data_num):
        golden = gold_list[data_id]
        pred = pred_list[data_id]
        with open(os.path.join(ref_dir, "%d_reference.txt" % (data_id)), 'w') as f:
            f.write(golden)
        with open(os.path.join(sum_dir, "%d_decoded.txt" % (data_id)), 'w') as f:
            f.write(pred)

    Rouge155_obj = Rouge155(stem=True, tmp=os.path.join(trick_result_dir, 'tmp'))
    scores = Rouge155_obj.evaluate_folder(sum_dir, ref_dir)
    with open(os.path.join(trick_result_dir, 'scores.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write(json.dumps(scores, indent=4))
    vis = vis_scores(scores)
    print(vis)


def work():
    parser = argparse.ArgumentParser()

    # path and old parameters
    parser.add_argument('--train_data_num', type=int, default=46773,
                        help='# of training set that can be used, -1 means all')
    parser.add_argument('--model-name', type=str, default='facebook/bart-base')
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--max-tgt-len', type=int, default=256)
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--beam_size', type=int, default=16)
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--data_path',
                        default='/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tf_idf',
                        help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--wikicatsum_dir',
                        default='/data1/tsq/wikicatsum',
                        help='origin dir of wikicatsum')
    parser.add_argument('--reload_ckpt', help='reload a checkpoint file')

    # parameters for tricks
    parser.add_argument('--max_beam_num', type=int, default=1)
    parser.add_argument("--duplicate_rate", type=float, default=0.7,
                        help="If the duplicat rate (compared with history) is large, we can discard the current sentence.")
    parser.add_argument('--strategy', type=str, nargs='+',
                        default=['fix_tokenization', 'remove_redundant', 'split_ref'],
                        help='how to use sorted sentences to generate final summary')

    args = parser.parse_args()
    # set pasth
    if args.model_name == 'unilm':
        args.save_dir = args.data_path
        log_dir = args.data_path
    else:
        model_clean_name = args.model_name.split("/")[-1]
        print(f"model_clean_name is {model_clean_name}")
        args.save_dir = os.path.join(args.data_path, f'{model_clean_name}_fine_tune')
        if args.train_data_num == -1:
            args.save_dir = os.path.join(args.save_dir,
                                         f'all_ml{args.max_len}_mtl{args.max_tgt_len}_me{args.max_epoch}')
        else:
            args.save_dir = os.path.join(args.save_dir,
                                         f'few_shot{args.train_data_num}_ml{args.max_len}_mtl{args.max_tgt_len}_me{args.max_epoch}')

        if args.reload_ckpt:
            args.save_dir = os.path.join(args.save_dir, 'reload')
        args.ckpt = os.path.join(args.save_dir, 'best_model.pt')
        ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
        data_base = os.path.splitext(os.path.basename(args.data_path))[0]
        log_dir = os.path.join(os.path.dirname(args.ckpt), 'test_%s_%s' % (data_base, ckpt_base))

    process_eval(args, log_dir)


if __name__ == '__main__':
    work()
