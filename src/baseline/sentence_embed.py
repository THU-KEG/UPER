import argparse
from tqdm import tqdm
import os
import pickle
import re
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize

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


def work():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    parser.add_argument('--data_dir', type=str, default="/data1/tsq/contrastive/clust_documents/",
                        help='dir of raw data (before clean noise)')

    args = parser.parse_args()
    sentence_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    for split in splits:
        tgt_dir = os.path.join(args.data_dir, args.category, 'raw')
        src_data_dir = os.path.join(tgt_dir, f'{split}_src')
        src_data_num = len(os.listdir(src_data_dir))

        data_id2sent_embeddings = []  # i: data_id, j: line_id, l[i][j]: sent_embedding
        for data_id in tqdm(range(src_data_num)):
            src_path = os.path.join(src_data_dir, f"{data_id}.txt")
            src_lines = open(src_path, 'r').readlines()
            sentences = []
            # get sentence str list
            for line_id, line in enumerate(src_lines):
                new_line = pat_letter.sub(' ', line).strip().lower()
                tokens = word_tokenize(new_line)
                sentences.append(' '.join(tokens))
            sent_embeddings = sentence_encoder.encode(sentences)
            data_id2sent_embeddings.append(sent_embeddings)

        # save
        sent_embeddings_path = os.path.join(tgt_dir, f'{split}_embeddings.pkl')
        with open(sent_embeddings_path, 'wb') as fout:
            pickle.dump(data_id2sent_embeddings, fout)


if __name__ == '__main__':
    work()
