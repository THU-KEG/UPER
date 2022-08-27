import json
import os
from tqdm import tqdm
import argparse
import torch
from transformers import BertTokenizer, RobertaTokenizer, BertForMaskedLM, BertConfig
from src.clust.pattern import select_clusts
from src.clust.gather import check_fout


def load_data(data_dir, category, split, object):
    """
    :return:  [ { title: str, docs: [document] } ]
    """
    raw_file_dir = os.path.join(data_dir, category, 'raw')
    title_file = os.path.join(raw_file_dir, f"{split}_titles.txt")
    raw_titles = open(title_file, 'r').readlines()
    titles = [title.strip() for title in raw_titles]
    data_num = len(titles)
    docs_dir = os.path.join(raw_file_dir, f'{split}_{object}')
    assert len(os.listdir(docs_dir)) == data_num
    test_data = []
    for i in range(data_num):
        src_file = os.path.join(docs_dir, f"{i}.txt")
        with open(src_file, 'r') as fin:
            raw_src_lines = fin.readlines()
            src_lines = [raw_src_line.strip() for raw_src_line in raw_src_lines]
            data_dict = {'title': titles[i], 'docs': src_lines}
            test_data.append(data_dict)
    print(f"load data from {raw_file_dir}")
    return test_data


def build_transformer_model(args):
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model)
    return model, tokenizer


def get_label_words(args, tokenizer):
    lead_sections = select_clusts(args)
    label_word_idxs = []
    for i, topic_words in enumerate(lead_sections):
        label_topic = []
        for j, topic_word in enumerate(topic_words):
            if len(topic_word.strip().split()) == 1:
                label_word = topic_word.strip().lower()
                label_topic.append(label_word)

        label_word_idxs.append(torch.LongTensor(tokenizer.convert_tokens_to_ids(label_topic)))
    print("label_word_idxs: ")
    print(label_word_idxs)
    print("lead sections:")
    print(lead_sections)
    return label_word_idxs


def get_topic_scores(mask_token_logits, label_word_idxs, args):
    if args.last_noisy:
        topic_scores = [0 for i in range(args.topic_num + 1)]
    else:
        topic_scores = [0 for i in range(args.topic_num)]
    for topic_id, label_words in enumerate(label_word_idxs):
        label_word_logits = torch.index_select(mask_token_logits, 0, label_words)
        if args.max_logit:
            topic_score = torch.max(label_word_logits).item()
        else:
            topic_score = torch.mean(label_word_logits).item()
        topic_scores[topic_id] = topic_score
    return topic_scores


def run(args):
    # get input
    test_data = load_data(args.data_dir, args.category, args.split, args.object)

    # generate feature logits for each document
    model, tokenizer = build_transformer_model(args)
    device = 'cuda'
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    mask_token_index = 5
    batch_size = args.batch_size
    # decide the label words
    label_word_idxs = get_label_words(args, tokenizer)
    save_root = os.path.join(args.data_dir, args.category, 'classify', args.split)
    if args.object == 'tgt':
        save_root = os.path.join(save_root, 'tgt')
    if args.last_noisy:
        save_dir = os.path.join(save_root, f'ls{args.lead_section_num}_t{args.topic_num}_noise')
    else:
        save_dir = os.path.join(save_root, f'ls{args.lead_section_num}_t{args.topic_num}')
    if args.max_logit:
        save_dir = os.path.join(save_dir, 'max_logit')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # data_id2doc_id2logits = []  # i: data_id, j: doc_id, k: tensor(shape: (vocab_size)) of mask token
    with torch.no_grad():
        for data_id, data_dict in tqdm(enumerate(test_data), total=len(test_data)):
            title = data_dict['title']
            docs = data_dict['docs']
            doc_num = len(docs)
            # doc_id2mask_token_logits = []
            save_path = os.path.join(save_dir, f"{data_id}_class.json")
            fout = check_fout(save_path)
            for doc_id in range(0, doc_num, batch_size):
                try:
                    batch_docs = docs[doc_id:doc_id + batch_size]
                except IndexError:
                    batch_docs = docs[doc_id:]
                sequences = [f"This document is about {tokenizer.mask_token} of {title} : {doc}" for doc in batch_docs]
                tokenize_res = tokenizer.batch_encode_plus(sequences, max_length=args.max_len, add_special_tokens=True,
                                                           return_tensors='pt', padding="max_length", truncation=True)
                input_ids = tokenize_res['input_ids'].to(device)
                output = model(input_ids, attention_mask=tokenize_res['attention_mask'].to(device),
                               token_type_ids=tokenize_res['token_type_ids'].to(device))
                token_logits = output.logits
                for i in range(len(sequences)):
                    mask_token_logits = token_logits[i, mask_token_index, :].cpu()  # shape: (vocab_size)
                    # doc_id2mask_token_logits.append(mask_token_logits)
                    topic_scores = get_topic_scores(mask_token_logits, label_word_idxs, args)
                    topic_num = topic_scores.index(max(topic_scores))
                    class_dict = {"doc_id": doc_id + i, "topic": topic_num, "scores": topic_scores}
                    fout.write(json.dumps(class_dict))
                    fout.write('\n')
            # data_id2doc_id2logits.append(doc_id2mask_token_logits)

            # torch.save(doc_id2mask_token_logits, save_path)


if __name__ == '__main__':
    # test_data_generator()
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='/data1/tsq/contrastive/clust_documents/')
    parser.add_argument('--split', type=str, default="test", choices=['train', 'valid', 'test'],
                        help='data split')
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])

    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    # data parameters
    parser.add_argument('--model', type=str, default='bert-large-cased')
    parser.add_argument('--lead_section_num', type=int, default=20,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    parser.add_argument('--last_noisy', action='store_true', help='whether use an extra topic as noise')
    parser.add_argument('--max_logit', action='store_true', help='whether use max or avg logit')

    parser.add_argument('--batch_size', type=int, default=4, help='# of topics')
    parser.add_argument('--max_len', type=int, default=64, help='# of topics')
    args = parser.parse_args()
    run(args)
