import argparse
from tqdm import tqdm
import os
import json
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from src.bart.preprocess import check_fout, mkdir

splits = ['test', 'train', 'valid']
pat_letter = re.compile(r'[^a-zA-Z \-]+')


def tokenize_lower(lines: list):
    """
    :param lines:
    :return: [[token]]
    """
    res = []
    for line in lines:
        new_line = pat_letter.sub(' ', line).strip().lower()
        tokens = word_tokenize(new_line)
        res.append(tokens)
    return res


def get_tf(title_token_list, freq_dict):
    tf = 0
    for title_token in title_token_list:
        tf += freq_dict[title_token]

    return tf


def work():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--data_dir', type=str, default="/data1/tsq/contrastive/clust_documents/",
                        help='dir of raw data (before clean noise)')

    args = parser.parse_args()

    for split in splits:
        tgt_dir = os.path.join(args.data_dir, args.category, 'raw')
        src_data_dir = os.path.join(tgt_dir, f'{split}_src')
        tf_dir = os.path.join(tgt_dir, f'{split}_tf')
        mkdir(tf_dir)
        title_path = os.path.join(tgt_dir, f"{split}_titles.txt")
        title_lines = open(title_path, 'r').readlines()
        title_tokens = tokenize_lower(title_lines)
        src_data_num = len(os.listdir(src_data_dir))
        assert len(title_tokens) == src_data_num

        data_id2freq_dicts = []  # i: data_id, j: line_id, l[i][j]: dict
        # data_id2tfs = []  # i: data_id, j: line_id, l[i][j]: term frequency
        for data_id in tqdm(range(src_data_num)):
            title_token_list = title_tokens[data_id]
            src_path = os.path.join(src_data_dir, f"{data_id}.txt")
            tf_path = os.path.join(tf_dir, f"{data_id}.txt")
            src_lines = open(src_path, 'r').readlines()
            freq_dicts = []
            # tfs
            tf_fout = check_fout(tf_path)
            for line_id, line in enumerate(src_lines):
                new_line = pat_letter.sub(' ', line).strip().lower()
                tokens = word_tokenize(new_line)
                freq_dict = nltk.FreqDist(tokens)
                freq_dicts.append(freq_dict)
                tf = get_tf(title_token_list, freq_dict)
                tf_fout.write(f"{tf}\n")
            data_id2freq_dicts.append(freq_dicts)
        # save
        freq_dict_path = os.path.join(tgt_dir, f'{split}_freq.pkl')
        with open(freq_dict_path, 'wb') as fout:
            pickle.dump(data_id2freq_dicts, fout)


if __name__ == '__main__':
    work()

    # text = "Azure-crowned hummingbird I'm yours"
    # pat_letter = re.compile(r'[^a-zA-Z \-]+')
    # new_text = pat_letter.sub(' ', text).strip().lower()
    # print(word_tokenize(new_text))
    # print(new_text)

    # texts = ["(nematoda: camallanidea) from",
    #          "view more articles from science .", "view"]
    #
    # Freq_dist_nltk = nltk.FreqDist(texts)
    # print(Freq_dist_nltk.freq("view"))
    # print(Freq_dist_nltk["view"])
