import os
import numpy as np
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
from transformers import BertTokenizer, RobertaTokenizer, BertForMaskedLM, BertConfig

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_classes = 32
maxlen = 128
batch_size = 8

# BERT base

config_path = 'data/pretrained/nezha/NEZHA-Base/bert_config.json'
checkpoint_path = 'data/pretrained/nezha/NEZHA-Base/model.ckpt-900000'
dict_path = 'data/pretrained/nezha/NEZHA-Base/vocab.txt'


def to_bert_input(token_idx, null_idx, segment_pos):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    if segment_pos > 0:
        segment_idx[:, segment_pos:] = token_idx[:, segment_pos:] > 0

    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    # token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


class PetModel(torch.nn.Module):

    def __init__(self, args, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        model_name = args.model_name_or_path
        self.NULL_IDX = self.tokenizer.pad_token_id
        self.positive_indexes = self.tokenizer.convert_tokens_to_ids(args.positive_words)
        self.negative_indexes = self.tokenizer.convert_tokens_to_ids(args.negative_words)

        self.bert_model = BertForMaskedLM.from_pretrained(model_name)
        self.context_len = args.max_context_length

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size))

        self.extra_token_embeddings = nn.Embedding(args.learnable_tokens + 1, self.bert_model.config.hidden_size)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def score_candidates(self, input_ids, input_flags, mlm_labels):
        input_ids, token_type_ids, attention_mask = to_bert_input(input_ids, self.NULL_IDX, self.context_len)

        # input_zeros = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
        # input()
        # masked_flags = torch.where(input_flags > 0, input_ids, input_zeros)
        # print(self.tokenizer.convert_ids_to_tokens(masked_flags[0]))
        # input()
        # masked_flags = torch.where(mlm_labels > 0, input_ids, input_zeros)
        # print(self.tokenizer.convert_ids_to_tokens(masked_flags[0]))
        # input()
        # print(input_flags[0])
        # input()

        # raw_embeddings = self.bert_model.embeddings.word_embeddings(input_ids)
        raw_embeddings = self.bert_model.bert.embeddings.word_embeddings(input_ids)
        new_token_embeddings = self.mlp(self.extra_token_embeddings.weight)

        new_embeddings = new_token_embeddings[input_flags]

        inputs_embeds = torch.where(input_flags.unsqueeze(-1) > 0, new_embeddings, raw_embeddings)
        results = self.bert_model(inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)

        logits = results.logits
        probs = torch.masked_select(logits, mlm_labels.unsqueeze(-1).bool()).view(input_ids.size(0), -1)

        # pos_probs = logits[:,mlm_labels,:][:,:,self.positive_indexes]
        # pos_logit = torch.sum(pos_probs, dim=(1,2)).unsqueeze(1)
        pos_probs = probs[:, self.positive_indexes]
        pos_logit = torch.sum(pos_probs, dim=1).unsqueeze(1)
        # neg_probs = logits[:,mlm_labels,:][:,:,self.negative_indexes]
        # neg_logit = torch.sum(neg_probs, dim=(1,2)).unsqueeze(1)
        neg_probs = probs[:, self.negative_indexes]
        neg_logit = torch.sum(neg_probs, dim=1).unsqueeze(1)
        # hidden_states = hidden_states[mlm_labels >= 0].view(input_ids.size(0), 1, -1)
        all_logits = torch.cat((neg_logit, pos_logit), dim=1)

        return all_logits


# pattern
pattern = '??????????????????????????????????????????:'
# tokenizer.encode?????????????????????cls?????????mask???index???+1
tokens = ["CLS"] + list(pattern)
print(tokens[14])
mask_idx = [14]

id2label = {
    0: '???',
    1: '???'
}

label2id = {v: k for k, v in id2label.items()}
print('label2id:', label2id)  # label2id: {'???': 0, '???': 1}
labels = list(id2label.values())
print('labels:', labels)  # labels: ['???', '???']
# labels???token??????ids,encode???????????????????????????cls????????????encode?????????tokens[1:-1]??????????????????cls???
label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])
print('label_ids:', label_ids)  # label_ids: [[ 856] [7770]]


class data_generator(DataGenerator):
    def __init__(self, prefix=False, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.prefix = prefix

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []
        # ??????query???reply
        for is_end, (q, r, label) in self.get_sample(shuffle):
            # ??????label??????????????????None
            label = int(label) if label is not None else None
            # ???label???????????????????????????
            if label is not None or self.prefix:
                q = pattern + q
            # ??????token_ids???segment_id
            token_ids, segment_ids = tokenizer.encode(q, r, maxlen=maxlen)
            # ????????????????????????
            if shuffle:
                # ??????????????????mask?????????mask???????????????, ??????????????????????????????
                source_tokens, target_tokens = random_masking(token_ids)
            else:
                # ?????????target_tokens?????????source_tokens
                source_tokens, target_tokens = token_ids[:], token_ids[:]
            # mask label
            if label is not None:
                # ???label?????????token????????????mlm??????????????????label????????????token
                label_ids = tokenizer.encode(id2label[label])[0][1:-1]
                # pattern = '??????????????????:'
                # mask_idx = [1]
                # ??????label_ids???????????????????????????????????????
                # mask_idx??????????????????label?????????????????????
                for m, lb in zip(mask_idx, label_ids):
                    # ???????????????????????????label????????????mask_id
                    # source_tokens[1] = mask_id
                    # ??????target_tokens[1] = label_id(?????????label?????????token_id)
                    # ??????????????????label?????????token?????????????????????
                    source_tokens[m] = tokenizer._token_mask_id
                    target_tokens[m] = lb
            elif self.prefix:
                # ???????????????mask_id?????????????????????????????????????????????token_id
                for i in mask_idx:
                    source_tokens[i] = tokenizer._token_mask_id
            # ????????????mlm?????????source_tokens,segment_ids,target_tokens
            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)

            if is_end or len(batch_token_ids) == self.batch_size:
                # ??????batch_size??????????????????yield??????
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_target_ids = pad_sequences(batch_target_ids)
                # batch_target_ids???????????????target???id
                yield [batch_token_ids, batch_segment_ids, batch_target_ids], None
                # ????????????batch????????????????????????
                batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []


def build_transformer_model(model, lead_section_num, topic_num, args):
    tokenizer = BertTokenizer.from_pretrained(args.model)
    return PetModel(args, tokenizer), tokenizer


def infer(test_data, args):
    model, tokenizer = build_transformer_model(args.model, args.lead_section_num, args.topic_num, args)
    # get input

    def evaluate(data):
        P, R, TP = 0., 0., 0.
        for d, _ in tqdm(data):
            x_true, y_true = d[:2], d[2]
            # ????????????????????????????????????label_ids?????????index???
            y_pred = predict(x_true)
            # ??????mask_idx?????????y -> ??????token -> ??????label??????index
            y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idx]])
            # print(y_true, y_pred)
            # ??????f1
            R += y_pred.sum()
            P += y_true.sum()
            TP += ((y_pred + y_true) > 1).sum()
        print(P, R, TP)
        pre = TP / R
        rec = TP / P
        return 2 * (pre * rec) / (pre + rec)

    def predict(x):
        if len(x) == 3:
            x = x[:2]
        # ??????mask_idx?????????output
        # todo:????????????model???????????????train_model????
        y_pred = model.predict(x)[:, mask_idx]
        # ??????????????????????????????
        # batch, 0,label_ids????????????, label_ids????????????????????????id??????????????????????????????
        y_pred = y_pred[:, 0, label_ids[:, 0]]
        # ?????????????????????label_ids???????????????????????????mlm?????????????????????????????????mlm???????????????????????????????????????label?????????
        y_pred = y_pred.argmax(axis=1)
        return y_pred

    class Evaluator(keras.callbacks.Callback):
        def __init__(self, valid_generator, best_pet_model_file="best_pet_model.weights"):
            self.best_acc = 0.
            self.valid_generator = valid_generator
            self.best_pet_model_file = best_pet_model_file

        def on_epoch_end(self, epoch, logs=None):
            acc = evaluate(self.valid_generator)
            if acc > self.best_acc:
                self.best_acc = acc
                self.model.save_weights(self.best_pet_model_file)
            print('acc :{}, best acc:{}'.format(acc, self.best_acc))

    def write_to_file(path, test_generator, test_data):
        preds = []
        # ??????????????????
        for x, _ in tqdm(test_generator):
            pred = predict(x)
            preds.extend(pred)

        # ????????????query???reply???????????????p?????????????????????
        ret = []
        for data, p in zip(test_data, preds):
            if data[2] is None:
                label = -1
            else:
                label = data[2]
            ret.append([data[0], data[1], str(label), str(p)])

        with open(path, 'w') as f:
            for r in ret:
                f.write('\t'.join(r) + '\n')

    evaluator = Evaluator(valid_generator, best_model_file)
    train_model.fit_generator(train_generator.generator(),
                              steps_per_epoch=len(train_generator),
                              epochs=10,
                              callbacks=[evaluator])

    train_model.load_weights(best_model_file)
    write_to_file(test_result_file, test_generator, test_data)


def load_pair_data(f, isshuffle=False):
    data = []
    df = pd.read_csv(f)
    if isshuffle:
        df = df.sample(frac=1.0, random_state=1234)
    columns = list(df.columns)
    if 'text_a' not in columns and 'query1' in columns:
        df.rename(columns={'query1': 'text_a', 'query2': 'text_b'}, inplace=True)
    for i in range(len(df)):
        can = df.iloc[i]
        text_a = can['text_a']
        text_b = can['text_b']
        if 'label' not in columns:
            label = None
        else:
            label = int(can['label'])
            if label == -1:
                label = None
        data.append([text_a, text_b, label])
    return data


def load_data(category, split):
    """
    :return: [text_a, text_b, label]
    ?????????????????????????????????
    """
    data_dir = f'/data1/tsq/contrastive/{category}/raw'
    test_file = data_dir + 'test.example_20200228.csv'
    test_data = load_pair_data(test_file)
    return test_data


def test_data_generator():
    data_dir = '../data/tianchi/'
    train_file = data_dir + 'train_20200228.csv'
    data = load_pair_data(train_file)
    train_generator = data_generator(data=data, batch_size=batch_size)
    for d in train_generator:
        print(d)
        break


def run(args):
    test_data = load_data(args.category, args.split)
    infer(test_data, args)


if __name__ == '__main__':
    # test_data_generator()
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--split', type=str, default="train", choices=['train', 'valid', 'test'],
                        help='data split')
    parser.add_argument('--start_id', type=int, default=0, help='start id of data, included')
    parser.add_argument('--end_id', type=int, default=46773, help='end id of data, not included')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'multi_news'])
    # data parameters
    parser.add_argument('--model', type=str, default='bert-large')
    parser.add_argument('--lead_section_num', type=int, default=10,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    args = parser.parse_args()
    run(args)
