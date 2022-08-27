import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BartTokenizer
from nltk.text import TextCollection
import nltk
from tqdm import tqdm
import pandas as pd
from nltk.corpus import stopwords
from spacy.lang.en import English
import os
import json
import re
from sklearn.preprocessing import StandardScaler
from src.clust.pattern import select_clusts, fill_pattern, get_inverse_pattern
from src.bart.preprocess import check_fout, mkdir

nlp = English()
nlp.add_pipe("sentencizer")
en_stop_words = stopwords.words('english')


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
    if tokenized:
        new_tf_idf_dir = os.path.join(new_dir, "origin", f"tokenized_{split}_tf_idf")
        new_norm_tf_idf_dir = os.path.join(new_dir, f"tokenized_{split}_tf_idf")
        new_src_dir = os.path.join(new_dir, f"tokenized_{split}_src")
        new_titles_path = os.path.join(new_dir, f"tokenized_{split}_titles.txt")
        new_pt_path = os.path.join(new_dir, f"tokenized_{split}_sentence_prefix_space.pt")
    else:
        new_tf_idf_dir = os.path.join(new_dir, "origin", f"{split}_tf_idf")
        new_norm_tf_idf_dir = os.path.join(new_dir, f"{split}_tf_idf")
        new_src_dir = os.path.join(new_dir, f"{split}_src")
        new_titles_path = os.path.join(new_dir, f"{split}_titles.txt")
        new_pt_path = os.path.join(new_dir, f"{split}_sentence_prefix_space.pt")

    if mode == 'output_tf_idf':
        mkdir(new_tf_idf_dir)
        output_tf_idf(new_tf_idf_dir, new_src_dir, new_titles_path)
        return
    elif mode == 'normalize_tf_idf':
        mkdir(new_norm_tf_idf_dir)
        normalize_tf_idf(new_norm_tf_idf_dir, new_tf_idf_dir, new_titles_path)
        return
    else:
        mkdir(new_src_dir)
        title_fout = check_fout(new_titles_path)

    sent2para_list = []  # list of dict, dict: key is sent_id, value is para_id, index of list is data_id
    para2sents_list = []  # list of dict,  dict: key is para_id, value is [sent_id], index of list is data_id
    sents_res_list = []  # list of tensor list
    with open(origin_doc_path, 'r') as fin:
        lines = fin.readlines()
        src_id = 0
        for i, line in tqdm(enumerate(lines), total=len(lines)):

            # read tf_idf score for each document
            # read source documents
            title_and_documents = json.loads(line.strip())
            src_path = os.path.join(new_src_dir, f"{src_id}.txt")

            document_list = title_and_documents["articles"]
            documents = [doc["text"] for doc in document_list]
            sents = []
            para_id = 0
            # output title
            title = title_and_documents["wiki_links"]
            if mode == 'sentencize':
                src_fout = check_fout(src_path)
                title_fout.write(json.dumps(title))
                title_fout.write('\n')

            # output documents
            rotate = para_sent_num - 1
            for j, document in enumerate(documents):
                # doc is not noise
                paragraph = document.replace("\n", "")
                para_sents = [str(sent).strip() for sent in nlp(paragraph).sents]
                para_len = len(paragraph.split())
                if len(para_sents) <= para_sent_num and para_len <= max_sent_len:
                    # this paragraph will be seen as a sentence
                    src_fout.write(paragraph)
                    src_fout.write("\n")
                    sents.append(prefix_space_tokenize(paragraph))
                else:
                    # write each sentences
                    out_sent = ""
                    out_len = 0
                    out_sent_num = 0
                    for sent in para_sents:
                        sent_len = len(sent.split())
                        if sent_len < 5:
                            continue
                        else:
                            out_len += sent_len
                            if sent_len < max_sent_len:
                                out_sent_num += 1
                                out_sent += sent

                        if out_sent_num % para_sent_num == rotate or out_len >= max_sent_len:
                            src_fout.write(out_sent)
                            src_fout.write("\n")
                            sents.append(prefix_space_tokenize(out_sent))
                            out_sent = ""
                            out_sent_num = 0
                            out_len = 0
                para_id += 1

            # tokenize result
            sents_res_list.append(sents)
            src_id += 1

    if mode == 'sentencize':
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


def get_document_list(new_src_dir, doc_id):
    src_path = os.path.join(new_src_dir, f"{doc_id}.txt")
    all_document_lst = []
    with open(src_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            document = line.lower()
            tokens = nltk.word_tokenize(document)
            filtered = [w for w in tokens if not w in en_stop_words]
            all_document_lst.append(filtered)
    return all_document_lst


def output_tf_idf(new_tf_idf_dir, new_src_dir, new_titles_path):
    with open(new_titles_path, 'r') as fin:
        lines = fin.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            title_lst = json.loads(line.strip())
            all_document_lst = get_document_list(new_src_dir, i)
            text_collection = TextCollection(all_document_lst)
            new_tf_idf_path = os.path.join(new_tf_idf_dir, f"{i}.txt")
            fout = check_fout(new_tf_idf_path)
            for document_text_lst in all_document_lst:
                tf_idf_lst = []
                if len(document_text_lst) > 0:
                    for wiki_link in title_lst:
                        entity_key_words = wiki_link.split("/")[-1].split("_")
                        full_word_tf_idf = 1
                        for key_word in entity_key_words:
                            word_tf_idf = text_collection.tf_idf(key_word.lower(), document_text_lst)
                            full_word_tf_idf *= word_tf_idf
                        tf_idf_lst.append(full_word_tf_idf)
                else:
                    # empty document, set tf-idf to -1
                    tf_idf_lst = [-1] * len(title_lst)
                fout.write(json.dumps(tf_idf_lst))
                fout.write("\n")


def normalize_tf_idf(tgt_norm_tf_idf_dir, src_tf_idf_dir, new_titles_path):
    scaler = StandardScaler()
    with open(new_titles_path, 'r') as fin:
        lines = fin.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            src_tf_idf_path = os.path.join(src_tf_idf_dir, f"{i}.txt")
            fout = check_fout(os.path.join(tgt_norm_tf_idf_dir, f"{i}.txt"))
            with open(src_tf_idf_path, 'r') as fin_tf_idf:
                lines_tf_idf = fin_tf_idf.readlines()
                doc2tf_idf = []
                for tf_idf_line in lines_tf_idf:
                    tf_idf_lst = json.loads(tf_idf_line)
                    doc2tf_idf.append(tf_idf_lst)
                if len(doc2tf_idf) == 0:
                    print(f"{i}th src has no sentence")
                else:
                    try:
                        normalized_tf_idfs = scaler.fit_transform(doc2tf_idf)
                    except ValueError:
                        # when there are no key words, they will raise ValueError, so no tf-idf
                        normalized_tf_idfs = [[0] for kk in range(len(doc2tf_idf))]
                    for normalized_tf_idf in normalized_tf_idfs:
                        fout.write(f"{sum(normalized_tf_idf)}\n")


def sentencize(args):
    patternList = getPatternList()
    splits = ['train', 'val', 'test']
    for split in splits:
        # for src, read once, output sentences and pt
        if args.object == 'src':
            # only_align =
            if args.mode == 'bart_len':
                count_bart_len(args.tgt_dir, split)
            else:
                src_sentencize_after_ignore(args.max_sent_len, args.para_sent_num, args.data_dir, f"{split}.jsonl",
                                            f"{split}.tfidf.src",
                                            args.tgt_dir, split, None, patternList,
                                            tokenized=False, mode=args.mode)
        else:
            # tgt_sentencize_after_ignore
            new_tgt_dir = os.path.join(args.tgt_dir, f"{split}_tgt")
            make_dirs(new_tgt_dir)
            origin_tgt_path = os.path.join(args.data_dir, f"{split}.target")
            with open(origin_tgt_path, 'r') as fin:
                lines = fin.readlines()
                tgt_id = 0
                for i, line in tqdm(enumerate(lines), total=len(lines)):
                    tgt_path = os.path.join(new_tgt_dir, f"{tgt_id}.txt")
                    tgt_fout = check_fout(tgt_path)
                    paragraph = line.strip()
                    para_sents = [str(sent).strip() for sent in nlp(paragraph).sents]
                    for sent in para_sents:
                        tgt_fout.write(f"{sent.strip()}\n")
                    tgt_id += 1


def generate_pattern(args):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    except ValueError:
        tokenizer = GPT2TokenizerFast.from_pretrained("/data1/tsq/WikiGen/pretrained_models/gpt2_large")

    splits = ['test', 'train', 'val']
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
                title = json.loads(raw_title.strip())
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
                        choices=['generate_pattern', 'sentencize', 'output_tf_idf', 'normalize_tf_idf', 'bart_len'],
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
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'wcep'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data/tsq/contrastive/clust_documents/wcep/raw/",
                        help='dir of raw data (before clean noise)')
    parser.add_argument('--tgt_dir', type=str, default="/data/tsq/contrastive/clust_documents/",
                        help='dir where processed data will go')
    args = parser.parse_args()
    if args.mode == 'sentencize' or args.mode == 'output_tf_idf' or args.mode == 'normalize_tf_idf' or args.mode == 'bart_len':
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

    if args.mode == 'sentencize' or args.mode == 'output_tf_idf' or args.mode == 'normalize_tf_idf' or args.mode == 'bart_len':
        sentencize(args)
    else:
        generate_pattern(args)


if __name__ == '__main__':
    work()
