import argparse
import random
import shutil
import os
import torch
from torch import nn, optim
import numpy as np

from utils import setup_logger, MetricLogger, strip_prefix_if_present
from data import TopicDataset
from model import BartDecodeModel
from validate import validate

import pdb
import traceback


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class GuruMeditation(torch.autograd.detect_anomaly):
    def __init__(self):
        super(GuruMeditation, self).__init__()

    def __enter__(self):
        super(GuruMeditation, self).__enter__()
        return self

    def __exit__(self, type, value, trace):
        super(GuruMeditation, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)

            print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            print("┃ Software Failure. Press left mouse button to continue ┃")
            print("┃        Guru Meditation 00000004, 0000AAC0             ┃")
            print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
            print(str(value))
            pdb.set_trace()


def train(args, logger):
    logger.info('[1] Loading data')

    train_loader = TopicDataset('train', args.data_path, shuffle=not (args.no_shuffle))
    # train_loader = TopicDataset(args.data_path, args.label_path, args.topic_path, 'train')
    valid_loader = TopicDataset('valid', args.data_path)
    test_loader = TopicDataset('test', args.data_path)
    logger.info('length of train/valid/test per gpu: %d/%d/%d' % (
        len(train_loader.data), len(valid_loader.data), len(test_loader.data)))

    logger.info('[2] Building model')
    device = torch.device('cuda')
    # encoder = BiGRUEncoder(args.dim_word, args.dim_h, args.num_layers, args.num_vocab, args.dropout).to(device)
    # document_decoder = DocumentDecoder(args.dim_h, args.num_topics + 1).to(device) # extra 'STOP' topic
    # sentence_decoder = SentenceGRUDecoder(args.num_vocab, args.dim_word, args.dim_h).to(device)
    # sentence_decoder = PointerGenerator(word_lookup=nn.Embedding(args.num_vocab, args.dim_word), 
    #                                     dim_word=args.dim_word, 
    #                                     dim_h=args.dim_h,
    #                                     num_layers=args.num_layers,
    #                                     num_vocab=args.num_vocab,
    #                                     dropout=args.dropout,
    #                                     min_dec_len=args.min_dec_len,
    #                                     max_dec_len=args.max_dec_len,
    #                                     beam_size=args.beam_size,
    #                                     is_coverage=args.is_coverage).to(device)

    model = BartDecodeModel(
        beam_size=args.beam_size,
        max_dec_len=args.max_dec_len,
    ).to(device)

    model_kwargs = {k: getattr(args, k) for k in
                    {'max_dec_len', 'beam_size', }
                    }

    # topic_generator = TopicGenerator(args.num_topics, device)

    logger.info('[3] Initializing word embeddings')
    # with torch.no_grad():
    #     weight = torch.tensor(train_loader.weight).float().to(device)
    #     print(weight)
    #     print('Shape of "weight":', weight.shape)
    #     print('Shape of "encoder.word_lookup.weight":', model.encoder.word_lookup.weight.shape)
    #     model.encoder.word_lookup.weight.set_(weight)
    #     model.sent_decoder.word_lookup.weight.set_(weight)
    #     model.word_lookup.weight.set_(weight)

    logger.info(model)
    # logger.info(list(model.named_parameters()))

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # if distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.local_rank], output_device=args.local_rank,
    #         # this should be removed if we update BatchNorm stats
    #         broadcast_buffers=False,
    #     )

    meters = MetricLogger(delimiter="  ")
    if args.reload_ckpt:
        assert args.is_coverage
        logger.info("Reload ckpt. Use coverage as Stage 2. Remember to use a small lr.")
        loaded = torch.load(args.reload_ckpt)
        loaded['state_dict'] = strip_prefix_if_present(loaded['state_dict'], prefix='module.')
        model.load_state_dict(loaded['state_dict'], strict=False)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.schedule_step, gamma=0.1)
    # score = validate(valid_loader, model, device, args.save_dir, train_loader.itow, fast=True)
    logger.info('[4] Start training......')
    for epoch_num in range(args.max_epoch):
        # scheduler.step()
        model.train()
        # for batch_iter, train_batch in enumerate(train_loader.gen_batch(required_index=18640)):
        for batch_iter, train_batch in enumerate(train_loader.gen_batch()):
            progress = epoch_num + batch_iter / train_loader.example_num

            # doc, summ, doc_len, summ_len, doc_label, summ_label = [a.to(device) for a in train_batch]
            input_ids, attention_mask, labels = [a.to(device) for a in train_batch]

            # topics, topic_len, topic_masks = topic_generator.gather_topics(doc, doc_len, doc_label)
            # label_loss, sentence_loss, coverage_loss = model(topics, topic_len, summ, summ_len, summ_label)
            loss = model(input_ids, attention_mask, labels)

            optimizer.zero_grad()

            # with GuruMeditation():
            #     losses.backward()

            loss.backward()

            if args.clip_value > 0:
                nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
            if args.clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters.update(loss=loss)

            # logger.info('Batch iter: %d, loss = %s' %(batch_iter, str(losses)))
            if (batch_iter + 1) % (train_loader.example_num // 100) == 0:

                logger.info(
                    meters.delimiter.join(
                        [
                            "progress: {prog:.2f}",
                            "{meters}",
                        ]
                    ).format(
                        prog=progress,
                        meters=str(meters),
                    )
                )
            #if (batch_iter + 1) % (train_loader.example_num // 200) == 0:
                #score = validate(valid_loader, model, device, args.save_dir,  fast=True)
                #logger.info("val")
                #logger.info(score)
                #save = {
                #   'kwargs': model_kwargs,
                #   'state_dict': model.state_dict(),
                #   'optimizer': optimizer.state_dict(),
                #}
                #torch.save(save,
                #   os.path.join(args.save_dir, 'model_progress%.2f_val%.3f.pt' % (progress, score['rouge_1_f_score'])))
                #if args.debug:
                #    logger.info("[debug] start valide at first")
                #    break

        score = validate(valid_loader, model, device, args.save_dir,  fast=True)
        logger.info("val")
        logger.info(score)
        save = {
            'kwargs': model_kwargs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step()

        torch.save(save,
                   os.path.join(args.save_dir, 'model_epoch%d_val%.3f.pt' % (epoch_num, score['rouge_1_f_score'])))


def parse_args():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--category', default='cross', choices=['animal', 'film', 'company','cross'])
    parser.add_argument('--data_path', default='/data1/tsq/cross_domain_twag/data/raw',
                        help='pickle file obtained by dataset dump or datadir for torchtext')

    parser.add_argument('--save_dir', default='/data1/tsq/WikiGen/bart/animal',
                        help='path to save checkpoints and logs')
    parser.add_argument('--reload_ckpt', help='reload a checkpoint file')

    # training parameters
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--schedule_step', type=int, nargs='+', default=[1])
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay rate per batch')
    parser.add_argument('--seed', type=int, default=666666, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optim', default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--clip_value', type=float, default=0.5)
    parser.add_argument('--clip_norm', type=float, default=2.0)
    parser.add_argument('--use_rl', action='store_true')

    # model parameters
    parser.add_argument('--model_type', default='abs', choices=['abs', 'ext', 'ext_abs'])
    parser.add_argument('--dim_word', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--dim_h', type=int, default=512, help='dimension of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')

    parser.add_argument('--debug', action='store_true')

    # decode parameters
    parser.add_argument('--min_dec_len', type=int, default=10)
    parser.add_argument('--max_dec_len', type=int, default=200)
    parser.add_argument('--max_dec_sent', type=int, default=15)
    parser.add_argument('--beam_size', type=int, default=4)

    # data parameters
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--fraction', type=float, default=1, help='fraction of training set reduction')

    args = parser.parse_args()

    #args.data_path = '/data1/tsq/wikicatsum/%s/bart_base_data' % args.category

    if (args.reload_ckpt) and (args.is_coverage):
        args.save_dir = '/data1/tsq/WikiGen/bart_2/%s' % args.category
    else:
        args.save_dir = '/data1/tsq/WikiGen/bart/%s_%s' % (args.category,args.max_dec_len)

    if args.reload_ckpt:
        # override model-related arguments when reloading
        model_arg_names = {'dim_word', 'dim_h', 'num_layers'}
        print(
            'reloading ckpt from %s, load its model arguments: [ %s ]' % (args.reload_ckpt, ', '.join(model_arg_names)))
        loaded = torch.load(args.reload_ckpt)
        model_kwargs = loaded['kwargs']
        for k in model_arg_names:
            setattr(args, k, model_kwargs[k])
    return args


def main():
    # dir preparation
    args = parse_args()
    # seed setting
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    logger = setup_logger("WikiGen", args.save_dir)

    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    train(args, logger)


if __name__ == '__main__':
    main()
