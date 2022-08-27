import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import os
import nltk
from difflib import SequenceMatcher
import re
from src.clust.score import make_dirs
from src.statistics.reproduce import call_rouge

stop_sentences = ['thank you', 'URLTOKEN', 'HTML', "The author's name is made", "browser",
                  "page", "email address", "cookie", "licence", "wiki"]

stop_patterns = [re.compile(s, re.I) for s in stop_sentences]
golden_origin = {
    'animal': '/data1/tsq/contrastive/rerank_documents/animal',
    'film': '/data1/tsq/contrastive/rerank_documents/film',
    'company': '/data1/tsq/contrastive/rerank_documents/company',
    'wcep': '/data/tsq/contrastive/clust_documents/wcep/raw/origin/'
}


def has_no_stop_words(sent):
    for stop_pattern in stop_patterns:
        searchObj = stop_pattern.search(sent)
        if searchObj:
            return False
    return True


def has_no_syntactic_err(sent):
    load_grammar = nltk.data.load('file:english_grammer.cfg')
    sent_split = sent.split()
    rd_parser = nltk.RecursiveDescentParser(load_grammar)
    for tree_struc in rd_parser.parse(sent_split):
        return True
    print(f"Wrong grammar: {sent}")
    return False


def strip_redundant_sentences(all_sents):
    """
    :param all_sents: [list of str] sentences
    :return: sentences after striping the redundant sentences
    """
    chosen_sents = []
    for sent in all_sents:
        flag = False
        for prev_sent in chosen_sents:
            s = SequenceMatcher(None, sent, prev_sent)
            ll = s.find_longest_match(0, len(sent), 0, len(prev_sent)).size

            if (ll * 2 >= len(sent)):
                flag = True
                break

        if not (flag):
            chosen_sents.append(sent.strip())

    return chosen_sents


def filt(sent_lines, strategy):
    if 'stop_sentences' in strategy:
        sent_lines = filter(has_no_stop_words, sent_lines)
    if 'filter_syntactic_err' in strategy:
        sent_lines = filter(has_no_syntactic_err, sent_lines)
    if 'remove_redundant' in strategy:
        sent_lines = strip_redundant_sentences(sent_lines)

    return sent_lines


def trunct(lines, max_token_num, keep_special_token):
    if keep_special_token:
        summary = " <EOP> ".join([line.strip() for line in lines])
    else:
        summary = "\n".join([line.strip() for line in lines])
    tokens = summary.split()
    trunct_tokens = tokens[:max_token_num]
    return ' '.join(trunct_tokens)


def gen_final_summary(args, tgt_dir, tf_policy, clust_policy, clust_num, clust_input_sent_num, clust_output_sent_num,
                      proportion, attenuation_coefficient, strategy, max_read_lines, max_token_num, start_id, end_id,
                      level, topic_score, prompt, rm_each_sent=False):
    # path
    if prompt == 'qa' or prompt == 'rouge':
        result_dir = os.path.join(tgt_dir, f"level_{level}")
        if level == 'topic':
            result_dir = os.path.join(result_dir, topic_score)
    else:
        result_dir = tgt_dir
    if tf_policy != 'no':
        result_dir = os.path.join(result_dir, tf_policy)
    if clust_policy != 'no':
        if args.last_noisy:
            result_dir = os.path.join(result_dir, clust_policy,
                                      f'k{clust_num}_i{clust_input_sent_num}_o{clust_output_sent_num}_{proportion}_noise')
        elif args.max_logit:
            result_dir = os.path.join(result_dir, clust_policy,
                                      f'k{clust_num}_i{clust_input_sent_num}_o{clust_output_sent_num}_{proportion}_max')
        else:
            result_dir = os.path.join(result_dir, clust_policy,
                                      f'k{clust_num}_i{clust_input_sent_num}_o{clust_output_sent_num}_{proportion}')
    if proportion not in ['free', 'tp']:
        result_dir = os.path.join(result_dir, f"{attenuation_coefficient}")
    sorted_dir = os.path.join(result_dir, "sorted")
    _strategy_name = "_".join([strate[:3] for strate in strategy])
    trunc_result_dir = os.path.join(result_dir, f"mrl{max_read_lines}_mtn{max_token_num}_{_strategy_name}")
    sum_dir = os.path.join(trunc_result_dir, "sum")
    make_dirs(sum_dir)
    for data_id in tqdm(range(start_id, end_id)):
        # read sorted sentences
        sentences_path = os.path.join(sorted_dir, f"{data_id}_sorted.txt")
        sent_lines = open(sentences_path, 'r').readlines()
        # lower, filter and truncate
        sent_lines = [sent.lower() for sent in sent_lines]
        final_lines = filt(sent_lines[:max_read_lines], strategy)
        final_summary = trunct(final_lines, max_token_num, args.keep_special_token)
        sum_path = os.path.join(sum_dir, f'{data_id}_decoded.txt')
        with open(sum_path, 'w') as fout:
            fout.write(final_summary)

    print(f"Summary is generated at {trunc_result_dir}")
    return trunc_result_dir


def copy_ref(result_dir, split, start_id, end_id, category):
    origin_dir = golden_origin[category]
    new_ref_dir = os.path.join(result_dir, 'ref')
    make_dirs(new_ref_dir)
    if category == 'wcep':
        origin_ref_path = f"/data/tsq/contrastive/clust_documents/wcep/raw/origin/{split}.target"
    else:
        origin_ref_path = os.path.join(origin_dir, f"{split}.target.tokenized")
    ref_lines = open(origin_ref_path, 'r').readlines()
    for i in range(start_id, end_id):
        new_ref_path = os.path.join(new_ref_dir, f"{i}_reference.txt")
        with open(new_ref_path, 'w') as fout:
            fout.write(ref_lines[i].strip())


def copy_ignore_ref(result_dir, split, start_id, end_id, category):
    origin_dir = f'/data1/tsq/wikicatsum/{category}'
    new_ref_dir = os.path.join(result_dir, 'ref')
    make_dirs(new_ref_dir)
    origin_ref_path = os.path.join(origin_dir, f"{split}.tgt")
    unignored_ref_lines = open(origin_ref_path, 'r').readlines()
    if split == 'train':
        ignore_path = os.path.join(origin_dir, f'ignoredIndices.log')
    else:
        ignore_path = os.path.join(origin_dir, f'{split}_ignoredIndices.log')
    ignore_idx_lines = open(ignore_path, 'r').readlines()
    ignore_idx = [int(l) for l in ignore_idx_lines]
    ref_lines = []
    for i in range(len(unignored_ref_lines)):
        if i in ignore_idx:
            pass
        else:
            ref_lines.append(unignored_ref_lines[i])
    assert len(ref_lines) == end_id - start_id
    for i in range(start_id, end_id):
        new_ref_path = os.path.join(new_ref_dir, f"{i}_reference.txt")
        with open(new_ref_path, 'w') as fout:
            fout.write(ref_lines[i].strip())


def work():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--tf', type=str, default="no",
                        choices=['no', 'not_zero', 'nz_concat', 'ws_0.5', 'ws_0.25', 'ws_0.75', 'ws_0.88', 'ws_0'],
                        # notice that score = ws * score - (1 - ws) * tf_idf, this suits for ppl score
                        # if is rouge score, then score = ws * score + (1 - ws) * tf_idf
                        help='how to use tf information')
    parser.add_argument('--clust', type=str, default="no",
                        choices=['no', 'k_means', 'zs_classify'],
                        help='ways of clustering')
    parser.add_argument('--clust_num', type=int, default=4,
                        help='how many cluster')
    parser.add_argument('--clust_input_sent_num', type=int, default=128,
                        help='how many sentences that can participate in clustering')
    parser.add_argument('--clust_output_sent_num', type=int, default=20,
                        help='if sent_num < clust_output_sent_num, we output them all and do not perform clustering')
    parser.add_argument('--strategy', type=str, nargs='+',
                        default=['filter_syntactic_err', 'stop_sentences', 'remove_redundant'],
                        help='how to use sorted sentences to generate final summary')
    parser.add_argument('--max_read_lines', type=int, default=20, help='# of lines that can be read')
    parser.add_argument('--max_token_num', type=int, default=200, help='# of tokens that can be read')
    parser.add_argument('--prompt', type=str, default="qa",
                        choices=['qa', 'inverse', 'none', 'rouge',
                                 'regression_lgb', 'regression_nn',
                                 'regression_lgb_r2_recall', 'regression_nn_r2_recall'],
                        help='ways of prompt')
    parser.add_argument('--rm_each_sent', action='store_true',
                        help='whether score paragraph and remove each sentence')
    parser.add_argument('--para_penal', type=float, default=0.5,
                        help='if use paragraph gain, the penalty of origin para ppl score')
    # data parameters
    parser.add_argument('--lead_section_num', type=int, default=10,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    parser.add_argument('--last_noisy', action='store_true', help='whether use an extra topic as noise')
    parser.add_argument('--max_logit', action='store_true', help='whether use max or avg logit')
    parser.add_argument('--proportion', type=str, default="free",
                        choices=['free', 'tp', 'ac', 'acr'],
                        help='how to decide the proportion of different topics')
    parser.add_argument('--attenuation_coefficient', type=float, default=0.9,
                        help='if use paragraph gain, the penalty of origin para ppl score')
    parser.add_argument('--addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')
    parser.add_argument('--keep_special_token', action='store_true', help='whether to keep special token')
    # sort parameters
    parser.add_argument('--level', type=str, default="all", choices=['all', 'topic', 'lead_section',
                                                                     'recall', 'precision', 'f1', 'r1_recall',
                                                                     'r2_recall', 'rl_recall'],
                        help='level of sort, first three are choices for qa prompt, last three are for rouge')

    parser.add_argument('--topic_score', type=str, default="min", choices=['avg', 'min'],
                        help='ways of calculating topic score using lead_section scores')
    # path
    parser.add_argument('--split', type=str, default="test", choices=['train', 'valid', 'test'],
                        help='data split')
    parser.add_argument('--start_id', type=int, default=0, help='start id of data, included')
    parser.add_argument('--end_id', type=int, default=1, help='end id of data, not included')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'wcep'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data/tsq/contrastive/clust_documents",
                        help='dir of raw data (after clean noise)')
    parser.add_argument('--tgt_dir', type=str, default="/data1/tsq/contrastive/clust_documents/animal/result/",
                        help='dir where processed data will go')
    args = parser.parse_args()

    args.sentences_dir = os.path.join(args.data_dir, args.category, 'raw', f"{args.split}_{args.object}")
    if args.prompt == 'inverse':
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'result',
                                    f'inverse_add{args.addition_pattern_num}', f"{args.split}_{args.object}")
    elif args.prompt == 'qa':
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'result',
                                    f'tn{args.topic_num}_lsn{args.lead_section_num}', f"{args.split}_{args.object}")
    else:
        # this is not a typical prompt, we just need rouge score as a upper bound
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'result', args.prompt, f"{args.split}_{args.object}")

    if args.rm_each_sent:
        args.tgt_dir = os.path.join(args.tgt_dir, f'{args.prompt}_para_penal{args.para_penal}')
    else:
        args.tgt_dir = os.path.join(args.tgt_dir, f'{args.prompt}_score')

    print(args)
    # generate final summary
    result_dir = gen_final_summary(args, args.tgt_dir, args.tf, args.clust,
                                   args.clust_num, args.clust_input_sent_num, args.clust_output_sent_num,
                                   args.proportion, args.attenuation_coefficient, args.strategy, args.max_read_lines,
                                   args.max_token_num,
                                   args.start_id, args.end_id, args.level, args.topic_score, args.prompt,
                                   args.rm_each_sent)
    # output ref and count rouge
    if args.keep_special_token:
        copy_ignore_ref(result_dir, args.split, args.start_id, args.end_id, args.category)
    else:
        copy_ref(result_dir, args.split, args.start_id, args.end_id, args.category)
    call_rouge(result_dir)


if __name__ == '__main__':
    work()
