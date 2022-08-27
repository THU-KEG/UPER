import argparse
import copy
import json
import logging

from tqdm import tqdm
import os
from src.bart.preprocess import work as preprocess
from src.bart.preprocess import check_fout
from src.clust.score import make_dirs


def extract_final_summary(args, ext_dir, split, strategy, max_read_lines, max_token_num, level,
                          topic_score, prompt, tgt_dir, add_title, title_path, tf_policy, clust_policy,
                          clust_num, clust_input_sent_num, clust_output_sent_num, proportion, attenuation_coefficient):
    # path
    if prompt == 'qa' or prompt == 'rouge':
        result_dir = os.path.join(ext_dir, f"level_{level}")
        if level == 'topic':
            result_dir = os.path.join(result_dir, topic_score)
    else:
        result_dir = ext_dir
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

    _strategy_name = "_".join([strate[:3] for strate in strategy])
    trunc_result_dir = os.path.join(result_dir, f"mrl{max_read_lines}_mtn{max_token_num}_{_strategy_name}")
    sum_dir = os.path.join(trunc_result_dir, "sum")
    ref_dir = os.path.join(trunc_result_dir, "ref")
    data_num = len(os.listdir(sum_dir))
    print(f"data_num is {data_num}")
    print(f"ref_dir has {len(os.listdir(ref_dir))}")
    assert len(os.listdir(ref_dir)) == data_num

    # read titles
    titles = open(title_path, 'r').readlines()[:data_num]
    assert len(titles) == data_num

    # output
    tgt_ext_path = os.path.join(tgt_dir, f"{split}.source")
    tgt_ref_path = os.path.join(tgt_dir, f"{split}.target")
    ext_fout = check_fout(tgt_ext_path)
    ref_fout = check_fout(tgt_ref_path)

    for data_id in tqdm(range(data_num)):
        # read sorted sentences
        sum_path = os.path.join(sum_dir, f'{data_id}_decoded.txt')
        with open(sum_path, 'r') as fin_sum:
            if add_title:
                if args.category == 'wcep':
                    title = json.loads(titles[data_id])
                    if len(title) > 0:
                        entity = " ".join(title[0].split("/")[-1].split("_"))
                        ext_fout.write(entity.strip())
                        ext_fout.write(' </s> ')
                else:
                    title = titles[data_id]
                    ext_fout.write(title.strip())
                    ext_fout.write(' </s> ')
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
    parser.add_argument('--add_title', action='store_true',
                        help='whether add tittle for each source document')
    parser.add_argument('--rm_each_sent', action='store_true',
                        help='whether score paragraph and remove each sentence')
    parser.add_argument('--para_penal', type=float, default=0.5,
                        help='if use paragraph gain, the penalty of origin para ppl score')
    # data parameters
    parser.add_argument('--last_noisy', action='store_true', help='whether use an extra topic as noise')
    parser.add_argument('--max_logit', action='store_true', help='whether use max or avg logit')
    parser.add_argument('--proportion', type=str, default="free",
                        choices=['free', 'tp', 'ac', 'acr'],
                        help='how to decide the proportion of different topics')
    parser.add_argument('--attenuation_coefficient', type=float, default=0.9,
                        help='if use paragraph gain, the penalty of origin para ppl score')
    parser.add_argument('--addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')

    parser.add_argument('--lead_section_num', type=int, default=10,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')

    # sort parameters
    parser.add_argument('--level', type=str, default="all", choices=['all', 'topic', 'lead_section',
                                                                     'recall', 'precision', 'f1', 'r1_recall',
                                                                     'r2_recall', 'rl_recall'],
                        help='level of sort, first three are choices for qa prompt, last three are for rouge')

    parser.add_argument('--topic_score', type=str, default="min", choices=['avg', 'min'],
                        help='ways of calculating topic score using lead_section scores')
    # path
    parser.add_argument('--split_mode', type=str, default="test_only",
                        choices=['3split', 'test_as_valid', 'test_only'],
                        help='how to use data split')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'wcep'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data/tsq/contrastive/clust_documents",
                        help='dir of raw data (after clean noise)')
    parser.add_argument('--bart_data_dir', type=str, default="/data1/tsq/contrastive/rerank_documents",
                        help='dir of raw data (i.e. titles)')
    parser.add_argument('--tgt_dir', type=str, default="/data1/tsq/contrastive/clust_documents/animal/extract",
                        help='dir where processed data will go')

    # Data parameters for bart
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--max-tgt-len', type=int, default=256, help="when training, we will truncate the target")
    parser.add_argument('--tokenizer-dir', type=str, default='facebook/bart-base')

    args = parser.parse_args()

    print(args)
    if args.split_mode == '3split':
        splits = ['train', 'val', 'test']
    elif args.split_mode == 'test_as_valid':
        splits = ['train', 'valid', 'test']
    else:
        # only test
        splits = ['test']

    if args.prompt == 'inverse':
        tgt_dir = os.path.join(args.data_dir, args.category, 'bart', args.split_mode,
                               f'inverse_add{args.addition_pattern_num}')
    elif args.prompt == 'qa':
        tgt_dir = os.path.join(args.data_dir, args.category, 'bart', args.split_mode,
                               f'tn{args.topic_num}_lsn{args.lead_section_num}')
    elif args.prompt == 'rouge':
        tgt_dir = os.path.join(args.data_dir, args.category, 'bart', args.split_mode,
                               f'rouge_{args.level}')
    else:
        tgt_dir = os.path.join(args.data_dir, args.category, 'bart', args.split_mode,
                               f'{args.prompt}')
    if args.tf != 'no':
        tgt_dir = os.path.join(tgt_dir, args.tf)
    if args.clust != 'no':
        if args.last_noisy:
            tgt_dir = os.path.join(tgt_dir, args.clust,
                                   f'k{args.clust_num}_i{args.clust_input_sent_num}_o{args.clust_output_sent_num}_ln')
        elif args.max_logit:
            tgt_dir = os.path.join(tgt_dir, args.clust,
                                   f'k{args.clust_num}_i{args.clust_input_sent_num}_o{args.clust_output_sent_num}_{args.proportion}_max')

        else:
            tgt_dir = os.path.join(tgt_dir, args.clust,
                                   f'k{args.clust_num}_i{args.clust_input_sent_num}_o{args.clust_output_sent_num}')
        if args.proportion not in ['free', 'tp']:
            tgt_dir = os.path.join(tgt_dir, f"{args.attenuation_coefficient}")

    if args.rm_each_sent:
        tgt_dir = os.path.join(tgt_dir, f'{args.prompt}_para_penal{args.para_penal}')
    elif args.add_title:
        tgt_dir = os.path.join(tgt_dir, f'{args.prompt}_add_title')
    else:
        tgt_dir = os.path.join(tgt_dir, f'{args.prompt}_score')

    if not os.path.exists(tgt_dir):
        make_dirs(tgt_dir)
    for split in splits:
        if args.split_mode == 'test_as_valid' and split == 'valid':
            source_split = 'test'
        else:
            source_split = split
        if args.prompt == 'inverse':
            args.res_dir = os.path.join(args.data_dir, args.category, 'result',
                                        f'inverse_add{args.addition_pattern_num}', f"{source_split}_{args.object}")
        elif args.prompt == 'qa':
            args.res_dir = os.path.join(args.data_dir, args.category, 'result',
                                        f'tn{args.topic_num}_lsn{args.lead_section_num}',
                                        f"{source_split}_{args.object}")
        else:
            # this is not a typical prompt, we just need rouge score as a upper bound
            args.res_dir = os.path.join(args.data_dir, args.category, 'result', args.prompt,
                                        f"{source_split}_{args.object}")

        if args.rm_each_sent:
            args.res_dir = os.path.join(args.res_dir, f'{args.prompt}_para_penal{args.para_penal}')
        else:
            args.res_dir = os.path.join(args.res_dir, f'{args.prompt}_score')
        if args.category == 'wcep':
            title_path = f"/data/tsq/contrastive/clust_documents/wcep/raw/{source_split}_titles.txt"
        else:
            title_path = os.path.join(args.bart_data_dir, args.category, f'{source_split}.source.tokenized')
        # generate final extracted summary for each split
        extract_final_summary(args, args.res_dir, split, args.strategy, args.max_read_lines, args.max_token_num,
                              args.level, args.topic_score, args.prompt,
                              tgt_dir, args.add_title, title_path, args.tf, args.clust, args.clust_num,
                              args.clust_input_sent_num, args.clust_output_sent_num, args.proportion,
                              args.attenuation_coefficient)

    # tokenize them by bart
    bart_args = copy.deepcopy(args)
    preprocess(bart_args, tgt_dir)

    print("[2]Finish bart preprocess at ", tgt_dir)


if __name__ == '__main__':
    work()
