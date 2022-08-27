import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import os
import json
# from compare_mt.rouge.rouge_scorer import RougeScorer
from src.bart.preprocess import check_fout

# all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
all_scorer = None
try:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
except ValueError:
    tokenizer = GPT2TokenizerFast.from_pretrained("/data/tsq/WikiGen/pretrained_models/gpt2_large")


def get_ppl(origin_input_ids, model, max_length, stride, device, fast=False):
    lls = []
    for i in range(0, origin_input_ids.size(1), stride):
        if fast:
            # in fact, if stride is very large, larger than the origin_input_ids.size(1), then fast mode won't work
            i = 0

        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, origin_input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = origin_input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

        if fast:
            break

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()


def calculate_ppl_score(sentences_dir, sentences_pt, pattern_dir, tgt_dir, split, model_id, start_id, end_id, prompt,
                        rm_each_sent):
    stride = 512
    max_length = 1024
    device = 'cuda'
    if prompt == 'rouge':
        model = None
    else:
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    # path
    if rm_each_sent:
        all_score_dir = os.path.join(tgt_dir, f'{prompt}_score_rm_each_sent')
    else:
        all_score_dir = os.path.join(tgt_dir, f'{prompt}_score')
    make_dirs(all_score_dir)

    # read src and pattern
    sents_res_list = torch.load(sentences_pt)
    if rm_each_sent:
        para2sents_json_path = os.path.join(sentences_dir, f"{split}_para2sents.json")
        with open(para2sents_json_path, 'r') as jin:
            para2sents_list = json.loads(jin.read())

    if prompt == 'rouge':
        data_id2patterns_token_res = open(pattern_dir, 'r').readlines()[start_id:end_id]
        # assert len(data_id2patterns_token_res) == end_id - start_id
    else:
        data_id2patterns_token_res = torch.load(os.path.join(pattern_dir, f"{split}_pattern.pt"))

    if rm_each_sent:
        # score every document(paragraph) with ppl score
        # for every sentences in a document, **remove** it then score the document, so we can get ppl score gain
        for data_id in tqdm(range(start_id, end_id)):
            all_score_path = os.path.join(all_score_dir, f"{data_id}_score.json")
            all_score_fout = check_fout(all_score_path)
            patterns_token_res = data_id2patterns_token_res[data_id]
            sents_res = sents_res_list[data_id]
            para2sents = para2sents_list[data_id]
            for para_id, sent_ids in para2sents.items():
                sent_id2para_socre = {}
                sent_num = len(sent_ids)
                # each paragraph will be scored for (1+sent_num) times
                origin_para_score = score_paragraph(prompt, sent_ids, sents_res, patterns_token_res, model,
                                                    max_length, stride, device)
                sent_id2para_socre['origin'] = origin_para_score
                if sent_num > 1:
                    for i in range(sent_num):
                        new_sent_ids = sent_ids[:i] + sent_ids[i + 1:]
                        rm_sent_score = score_paragraph(prompt, new_sent_ids, sents_res, patterns_token_res, model,
                                                        max_length, stride, device)
                        sent_id2para_socre[sent_ids[i]] = rm_sent_score

                # output sent_id2para_socre
                all_score_fout.write(json.dumps(sent_id2para_socre))
                all_score_fout.write('\n')
    else:
        # score every sentence with ppl score
        for data_id in tqdm(range(start_id, end_id)):
            sents_res = sents_res_list[data_id]
            all_score_path = os.path.join(all_score_dir, f"{data_id}_score.txt")
            all_score_fout = check_fout(all_score_path)
            for sent_id, sent_res in enumerate(sents_res):

                if prompt == 'qa' or prompt == 'inverse' or prompt == 'rouge' or prompt == 'fast_inverse':
                    patterns_token_res = data_id2patterns_token_res[data_id]
                    score = score_sentence(prompt, sent_res, patterns_token_res, model, max_length, stride, device)
                    # output all scores
                    all_score_fout.write(json.dumps(score))
                    all_score_fout.write('\n')
                else:
                    score = score_sentence(prompt, sent_res, None, model, max_length, stride, device)
                    all_score_fout.write("%f\n" % score)


def score_paragraph(prompt, sent_ids, sents_res, patterns_token_res, model, max_length, stride, device):
    para_sents = [sents_res[sent_id] for sent_id in sent_ids]
    para_input_ids = torch.cat(para_sents, 1)
    return score_sentence(prompt, para_input_ids, patterns_token_res, model, max_length, stride, device)


def score_sentence(prompt, sent_res, patterns_token_res, model, max_length, stride, device):
    if prompt == 'qa':
        topic_scores = []  # i: topic_id , j: section_id
        for topic_patterns in patterns_token_res:
            ppl_scores = []
            for pattern in topic_patterns:
                # calculate once
                # origin_input_ids = torch.cat((pattern, sent_res), 1)[:, :max_length]
                origin_input_ids = torch.cat((pattern, sent_res), 1)
                ppl = get_ppl(origin_input_ids, model, max_length, stride, device)
                ppl_scores.append(ppl)

            topic_scores.append(ppl_scores)
        return topic_scores
    elif prompt == 'inverse':
        ppl_scores = []
        for pattern in patterns_token_res:
            # calculate once
            # origin_input_ids = torch.cat((pattern, sent_res), 1)[:, :max_length]
            origin_input_ids = torch.cat((sent_res, pattern), 1)
            ppl = get_ppl(origin_input_ids, model, max_length, stride, device)
            ppl_scores.append(ppl)
        return ppl_scores
    elif prompt == 'fast_inverse':
        ppl_scores = []
        for pattern in patterns_token_res:
            # calculate once
            # origin_input_ids = torch.cat((pattern, sent_res), 1)[:, :max_length]
            origin_input_ids = torch.cat((sent_res, pattern), 1)
            ppl = get_ppl(origin_input_ids, model, max_length, stride, device, fast=True)
            ppl_scores.append(ppl)
        return ppl_scores

    elif prompt == 'rouge':
        # use rouge score, so the patterns_token_res is the ref summary string
        score = all_scorer.score(patterns_token_res.strip(), tokenizer.decode(sent_res[0]).strip())
        rouge_scores = [
            score["rouge1"].recall, score["rouge1"].precision, score["rouge1"].fmeasure,
            score["rouge2"].recall, score["rouge2"].precision, score["rouge2"].fmeasure,
            score["rougeLsum"].recall, score["rougeLsum"].precision, score["rougeLsum"].fmeasure,
        ]
        return rouge_scores
    else:
        # prompt is none
        # origin_input_ids = sent_res[:, :max_length]
        origin_input_ids = sent_res
        ppl = get_ppl(origin_input_ids, model, max_length, stride, device)
        return ppl


def make_dirs(dir):
    if not (os.path.exists(dir)):
        os.makedirs(dir)


def work():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_id', type=str, default="gpt2-large",
                        help='model of scorer')
    parser.add_argument('--prompt', type=str, default="fast_inverse",
                        choices=['qa', 'inverse', 'fast_inverse', 'none', 'rouge'],
                        help='ways of prompt')
    parser.add_argument('--rm_each_sent', action='store_true',
                        help='whether score paragraph and remove each sentence')
    # data parameters
    parser.add_argument('--lead_section_num', type=int, default=10,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    parser.add_argument('--addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')

    # path
    parser.add_argument('--split', type=str, default="test", choices=['train', 'valid', 'val', 'test'],
                        help='data split')
    parser.add_argument('--start_id', type=int, default=0, help='start id of data, included')
    parser.add_argument('--end_id', type=int, default=2573, help='end id of data, not included')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'wcep'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data/tsq/contrastive/clust_documents",
                        help='dir of raw data (after clean noise)')
    parser.add_argument('--bart_data_dir', type=str, default="/data/tsq/contrastive/rerank_documents",
                        help='dir of raw data ')
    parser.add_argument('--tgt_dir', type=str, default="/data/tsq/contrastive/clust_documents/wcep/score/",
                        help='dir where processed data will go')
    args = parser.parse_args()

    args.sentences_dir = os.path.join(args.data_dir, args.category, 'raw')
    if args.object == 'src':
        args.sentences_pt = os.path.join(args.data_dir, args.category, 'raw', f"{args.split}_sentence_prefix_space.pt")

    if args.prompt == 'inverse' or args.prompt == 'fast_inverse':
        args.pattern_dir = os.path.join(args.data_dir, args.category, 'patterns',
                                        f'inverse_add{args.addition_pattern_num}')
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'score',
                                    f'{args.prompt}_add{args.addition_pattern_num}', f"{args.split}_{args.object}")
    elif args.prompt == 'rouge':
        # this is not a typical prompt, we just need rouge score as a upper bound
        args.pattern_dir = os.path.join(args.bart_data_dir, args.category, f'{args.split}.target.tokenized')
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'score', 'rouge', f"{args.split}_{args.object}")
    else:
        args.pattern_dir = os.path.join(args.data_dir, args.category, 'patterns',
                                        f'tn{args.topic_num}_lsn{args.lead_section_num}')
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'score',
                                    f'tn{args.topic_num}_lsn{args.lead_section_num}', f"{args.split}_{args.object}")

    make_dirs(args.tgt_dir)

    # titles_path = os.path.join(args.data_dir, args.category, 'raw', f'{args.split}_titles.txt')

    calculate_ppl_score(args.sentences_dir, args.sentences_pt, args.pattern_dir, args.tgt_dir, args.split,
                        args.model_id, args.start_id,
                        args.end_id, args.prompt, args.rm_each_sent)


if __name__ == '__main__':
    work()
