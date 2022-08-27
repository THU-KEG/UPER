import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BartTokenizer
from tqdm import tqdm
import pandas as pd
from spacy.lang.en import English
import os
import json
import re
from sklearn.preprocessing import StandardScaler
from src.clust.pattern import select_clusts, fill_pattern, get_inverse_pattern
from src.bart.preprocess import check_fout, mkdir

nlp = English()
nlp.add_pipe("sentencizer")


def make_dirs(dir):
    if not (os.path.exists(dir)):
        os.makedirs(dir)

def getPatternList():
    patternList = []

    patternList.append(re.compile(r'html'))
    patternList.append(re.compile(r'w3c'))
    patternList.append(re.compile(r'urltoken'))
    patternList.append(re.compile(r'cookies'))
    patternList.append(re.compile(r'href'))

    patternList.append(re.compile(r'\[ details \]'))
    patternList.append(re.compile(r'automatically generated'))
    patternList.append(re.compile(r'\[ maps \]'))

    patternList.append(re.compile(r'copyright'))
    patternList.append(re.compile(r'©'))

    patternList.append(re.compile(r'\W\s\d+\spp'))

    return patternList


def src_sentencize_after_ignore(max_sent_len, para_sent_num, origin_dir, origin_doc_name, origin_tf_idf_name,
                                new_dir, split, ignore_list, patternList, tokenized, mode='sentencize'):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    except ValueError:
        tokenizer = GPT2TokenizerFast.from_pretrained("/data1/tsq/WikiGen/pretrained_models/gpt2_large")

    def prefix_space_tokenize(sentence):
        # Token indices sequence length is longer than the specified maximum sequence length for this model
        # (1541 > 1024). Running this sequence through the model will result in indexing errors
        tokenizer_res = tokenizer(" " + sentence, return_tensors='pt').input_ids
        if tokenizer_res.shape[1] > 1024:
            print("Sentence too long for gpt2: ", sentence)
            tokenizer_res = tokenizer_res[:, :1024]

        return tokenizer_res

    origin_doc_path = os.path.join(origin_dir, origin_doc_name)
    origin_tf_idf_path = os.path.join(origin_dir, origin_tf_idf_name)
    if tokenized:
        new_tf_idf_dir = os.path.join(new_dir, f"tokenized_{split}_tf_idf")
        new_src_dir = os.path.join(new_dir, f"tokenized_{split}_src")
        new_titles_path = os.path.join(new_dir, f"tokenized_{split}_titles.txt")
        new_pt_path = os.path.join(new_dir, f"tokenized_{split}_sentence_prefix_space.pt")
    else:
        new_tf_idf_dir = os.path.join(new_dir, f"{split}_tf_idf")
        new_src_dir = os.path.join(new_dir, f"{split}_src")
        new_titles_path = os.path.join(new_dir, f"{split}_titles.txt")
        new_pt_path = os.path.join(new_dir, f"{split}_sentence_prefix_space.pt")

    if mode == 'align_sent_para':
        sent2para_list_json_path = os.path.join(new_dir, f"{split}_sent2para.json")
        para2sents_list_json_path = os.path.join(new_dir, f"{split}_para2sents.json")
    elif mode == 'output_tf_idf':
        mkdir(new_tf_idf_dir)
    else:
        mkdir(new_src_dir)
        title_fout = check_fout(new_titles_path)

    sent2para_list = []  # list of dict, dict: key is sent_id, value is para_id, index of list is data_id
    para2sents_list = []  # list of dict,  dict: key is para_id, value is [sent_id], index of list is data_id
    sents_res_list = []  # list of tensor list
    tf_idf_lines = open(origin_tf_idf_path, 'r').readlines()
    scaler = StandardScaler()
    with open(origin_doc_path, 'r') as fin:
        lines = fin.readlines()
        assert len(lines) == len(tf_idf_lines)
        src_id = 0
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            if i in ignore_list:
                continue
            else:
                # read tf_idf score for each document
                tf_idfs = tf_idf_lines[i].split('<EOP>')
                tf_idf_path = os.path.join(new_tf_idf_dir, f"{src_id}.txt")
                tf_idf_for_sentences = []
                # read source documents
                title_and_documents = line.split("<EOT>")
                src_path = os.path.join(new_src_dir, f"{src_id}.txt")
                sent2para = {}
                para2sents = {}

                documents_str = title_and_documents[1]
                documents = documents_str.split("<EOP>")
                sents = []
                para_id = 0
                sent_id = 0
                # output title
                title = title_and_documents[0]
                if mode == 'sentencize':
                    src_fout = check_fout(src_path)
                    title_fout.write(title.strip())
                    title_fout.write('\n')
                elif mode == 'output_tf_idf':
                    tf_idf_fout = check_fout(tf_idf_path)
                    try:
                        tf_idf_scores = [float(tf_idfs[j].strip()) for j in range(len(documents))]
                    except ValueError:
                        print(f"{src_id}th src has no tf_idf_scores")
                        src_id += 1
                        continue
                # output documents
                for j, document in enumerate(documents):
                    isNoise = False
                    for pattern in patternList:
                        searchObj = pattern.search(document)
                        if searchObj:
                            # document is noise
                            isNoise = True
                            break
                    if not isNoise:
                        # doc is not noise
                        paragraph = document.strip()
                        para_sents = [str(sent).strip() for sent in nlp(paragraph).sents]
                        para_len = len(paragraph.split())
                        if len(para_sents) <= para_sent_num and para_len <= max_sent_len:
                            # this paragraph will be seen as a sentence
                            if mode == 'align_sent_para':
                                sent2para[sent_id] = para_id
                                para2sents[para_id] = [sent_id]
                                sent_id += 1
                            elif mode == 'output_tf_idf':
                                tf_idf_for_sentences.append([tf_idf_scores[j]])
                            else:
                                src_fout.write(paragraph)
                                src_fout.write("\n")
                                sents.append(prefix_space_tokenize(paragraph))
                        else:
                            # write each sentences
                            for sent in para_sents:
                                sent_len = len(sent.split())
                                if sent_len < 5:
                                    continue
                                if sent_len < max_sent_len:
                                    if mode == 'align_sent_para':
                                        sent2para[sent_id] = para_id
                                        try:
                                            para2sents[para_id].append(sent_id)
                                        except KeyError:
                                            para2sents[para_id] = [sent_id]
                                        sent_id += 1
                                    elif mode == 'output_tf_idf':
                                        tf_idf_for_sentences.append([tf_idf_scores[j]])
                                    else:
                                        src_fout.write(sent)
                                        src_fout.write("\n")
                                        sents.append(prefix_space_tokenize(sent))
                        para_id += 1

                # output normalized tf_idf scores of each sentence
                if mode == 'output_tf_idf':
                    if len(tf_idf_for_sentences) == 0:
                        print(f"{src_id}th src has no sentence")
                    else:
                        normalized_tf_idfs = scaler.fit_transform(tf_idf_for_sentences)
                        for normalized_tf_idf in normalized_tf_idfs:
                            tf_idf_fout.write(f"{normalized_tf_idf[0]}\n")
                # tokenize result
                sents_res_list.append(sents)
                sent2para_list.append(sent2para)
                para2sents_list.append(para2sents)
                src_id += 1

    if mode == 'align_sent_para':
        with open(sent2para_list_json_path, 'w') as json_fout1:
            json_fout1.write(json.dumps(sent2para_list))
        with open(para2sents_list_json_path, 'w') as json_fout2:
            json_fout2.write(json.dumps(para2sents_list))
    elif mode == 'sentencize':
        # output sentence_prefix_space.pt
        # list of tensor list
        # i: data_id, j: sentence_id
        torch.save(sents_res_list, new_pt_path)


def count_bart_len(new_dir, split):
    try:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    except ValueError:
        tokenizer = BartTokenizer.from_pretrained('/data1/tsq/WikiGen/pretrained_models/bart_base')
    src_dir = os.path.join(new_dir, f"{split}_src")
    bart_len_dir = os.path.join(new_dir, f"{split}_bart_len")
    make_dirs(bart_len_dir)
    data_num = len(os.listdir(src_dir))
    print(data_num)
    for data_id in range(data_num):
        src_path = os.path.join(src_dir, f"{data_id}.txt")
        bart_len_path = os.path.join(bart_len_dir, f"{data_id}.txt")
        bl_fout = check_fout(bart_len_path)
        doc_lines = open(src_path, 'r').readlines()
        for doc_line in doc_lines:
            doc = doc_line.strip()
            tokenize_res = tokenizer.batch_encode_plus([doc], max_length=400, return_tensors="pt", truncation=True)
            input_ids = tokenize_res['input_ids']
            # -2 because of bos_token': '<s>', 'eos_token': '</s>'
            true_len = input_ids.shape[1] - 2
            bl_fout.write(f"{true_len}\n")
            # print(tokenizer.special_tokens_map)
            # print(tokenizer.all_special_ids)
            # print(input_ids.shape)
            # print(input_ids.shape[1])
            # quit()


def sentencize(args):
    patternList = getPatternList()
    splits = ['train', 'valid', 'test']
    for split in splits:
        if split == 'train':
            ignore_filename = os.path.join(args.data_dir, 'ignoredIndices.log')
        else:
            ignore_filename = os.path.join(args.data_dir, '%s_ignoredIndices.log' % split)

        ignore_list = set()
        if os.path.exists(ignore_filename):
            with open(ignore_filename, 'r') as fin:
                for line in fin:
                    ignore_list.add(int(line.strip()))
        else:
            print("File {} does not exist, maybe you have ignored before".format(ignore_filename))

        # for src, read once, output sentences and pt
        if args.object == 'src':
            # only_align =
            if args.mode == 'bart_len':
                count_bart_len(args.tgt_dir, split)
            else:
                src_sentencize_after_ignore(args.max_sent_len, args.para_sent_num, args.data_dir, f"{split}.raw.src",
                                            f"{split}.tfidf.src",
                                            args.tgt_dir, split, ignore_list, patternList,
                                            tokenized=False, mode=args.mode)
        else:
            # tgt_sentencize_after_ignore
            new_tgt_dir = os.path.join(args.tgt_dir, f"{split}_tgt")
            make_dirs(new_tgt_dir)
            origin_tgt_path = os.path.join(args.data_dir, f"{split}.raw.tgt")
            with open(origin_tgt_path, 'r') as fin:
                lines = fin.readlines()
                tgt_id = 0
                for i, line in tqdm(enumerate(lines), total=len(lines)):
                    if i in ignore_list:
                        continue
                    else:
                        tgt_path = os.path.join(new_tgt_dir, f"{tgt_id}.txt")
                        tgt_fout = check_fout(tgt_path)
                        paragraph = line.strip()
                        para_sents = [str(sent).strip() for sent in nlp(paragraph).sents]
                        for sent in para_sents:
                            if len(sent.split()) < 5:
                                continue
                            tgt_fout.write(f"{sent.strip()}\n")
                        tgt_id += 1


def generate_pattern(args):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    except ValueError:
        tokenizer = GPT2TokenizerFast.from_pretrained("/data1/tsq/WikiGen/pretrained_models/gpt2_large")

    splits = ['test', 'train', 'valid']
    for split in splits:
        # read titles
        titles_path = os.path.join(args.title_dir, f"{split}_titles.txt")
        with open(titles_path, 'r') as fin:
            titles = fin.readlines()

        # concat title and pattern and tokenize them
        patterns_dir = os.path.join(args.tgt_dir, split)
        make_dirs(patterns_dir)
        if args.prompt == 'inverse':
            data_id2patterns_token_res = []
            for data_id, raw_title in tqdm(enumerate(titles)):
                title = raw_title.strip()
                patterns = get_inverse_pattern(args, title)
                patterns_token_res = []
                for pattern in patterns:
                    token_res = tokenizer(pattern, return_tensors='pt').input_ids
                    patterns_token_res.append(token_res)
                data_id2patterns_token_res.append(patterns_token_res)
                patterns_path = os.path.join(patterns_dir, f"{data_id}_pattern.json")
                with open(patterns_path, 'w') as json_fout:
                    json_fout.write(json.dumps(patterns))
            # output sentence_prefix_space.pt
            new_pt_path = os.path.join(args.tgt_dir, f"{split}_pattern.pt")
            # list of list of tensor
            # i: data_id, j: pattern_id
            torch.save(data_id2patterns_token_res, new_pt_path)
        else:
            # prompt is qa
            lead_section_clusts = select_clusts(args)
            data_id2topic_patterns_token_res_list = []
            for data_id, raw_title in tqdm(enumerate(titles)):
                title = raw_title.strip()
                patterns_path = os.path.join(patterns_dir, f"{data_id}_pattern.json")
                patterns = []
                topic_patterns_token_res_list = []
                for topic_lead_sections in lead_section_clusts:
                    topic_patterns = []
                    topic_patterns_token_res = []
                    for topic_lead_section in topic_lead_sections:
                        section_pattern = fill_pattern(topic_lead_section, title)
                        token_res = tokenizer(section_pattern, return_tensors='pt').input_ids
                        topic_patterns.append(section_pattern)
                        topic_patterns_token_res.append(token_res)
                    patterns.append(topic_patterns)
                    topic_patterns_token_res_list.append(topic_patterns_token_res)

                with open(patterns_path, 'w') as json_fout:
                    json_fout.write(json.dumps(patterns))
                data_id2topic_patterns_token_res_list.append(topic_patterns_token_res_list)

            # output sentence_prefix_space.pt
            new_pt_path = os.path.join(args.tgt_dir, f"{split}_pattern.pt")
            # list of list of tensor list
            # i: data_id, j: topic_id, k: pattern tensor id
            torch.save(data_id2topic_patterns_token_res_list, new_pt_path)


def work():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--mode', type=str, default="bart_len",
                        choices=['generate_pattern', 'sentencize', 'align_sent_para', 'output_tf_idf', 'bart_len'],
                        help='task of preprocess')
    parser.add_argument('--prompt', type=str, default="qa", choices=['qa', 'inverse', 'none'],
                        help='ways of prompt')
    # data parameters
    parser.add_argument('--lead_section_num', type=int, default=10,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    parser.add_argument('--addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')
    parser.add_argument('--max_sent_len', type=int, default=64, help='max #token of a sentence')
    parser.add_argument('--para_sent_num', type=int, default=3,
                        help='for one paragraph, how many sentences can be seen as one sentence')

    # path
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data1/tsq/wikicatsum/",
                        help='dir of raw data (before clean noise)')
    parser.add_argument('--tgt_dir', type=str, default="/data1/tsq/contrastive/clust_documents/",
                        help='dir where processed data will go')
    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.category)
    if args.mode == 'sentencize' or args.mode == 'align_sent_para' or args.mode == 'output_tf_idf' or args.mode == 'bart_len':
        args.tgt_dir = os.path.join(args.tgt_dir, args.category, 'raw')
    else:
        args.title_dir = os.path.join(args.tgt_dir, args.category, 'raw')
        if args.prompt == 'inverse':
            args.tgt_dir = os.path.join(args.tgt_dir, args.category, 'patterns',
                                        f'inverse_add{args.addition_pattern_num}')
        else:
            args.tgt_dir = os.path.join(args.tgt_dir, args.category, 'patterns',
                                        f'tn{args.topic_num}_lsn{args.lead_section_num}')

    if not os.path.exists(args.tgt_dir):
        os.makedirs(args.tgt_dir)

    if args.mode == 'sentencize' or args.mode == 'align_sent_para' or args.mode == 'output_tf_idf' or args.mode == 'bart_len':
        sentencize(args)
    else:
        generate_pattern(args)


if __name__ == '__main__':
    work()
