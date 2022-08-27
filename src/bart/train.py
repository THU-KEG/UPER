import argparse
import random
import shutil
import os
import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm

from src.bart.utils import setup_logger, MetricLogger, strip_prefix_if_present
from src.bart.data import TopicDataset
from src.bart.model import BartDecodeModel
from src.bart.validate import validate, call_validate

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


def avg(score):
    f_score = score['rouge_1_f_score'] + score['rouge_2_f_score'] + score['rouge_l_f_score']
    return f_score / 3


def train(args, logger):
    logger.info('[1] Loading data')
    train_loader = TopicDataset('train', args.data_path, shuffle=not args.no_shuffle, max_len=args.max_len,
                                max_target_len=args.max_tgt_len, use_data_num=args.train_data_num)
    valid_loader = TopicDataset('valid', args.data_path, max_len=args.max_len, max_target_len=args.max_tgt_len)
    test_loader = TopicDataset('test', args.data_path, max_len=args.max_len, max_target_len=args.max_tgt_len)
    logger.info('length of train/valid/test per gpu: %d/%d/%d' % (
        len(train_loader.data), len(valid_loader.data), len(test_loader.data)))

    logger.info('[2] Building model')
    device = torch.device('cuda')

    model = BartDecodeModel(args.model_name).to(device)

    model_kwargs = {k: getattr(args, k) for k in
                    {'max_dec_len', 'beam_size', 'min_dec_len'}
                    }

    logger.info('[3] Initializing word embeddings')
    logger.info(model)

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    meters = MetricLogger(delimiter="  ")
    if args.reload_ckpt:
        logger.info("Reload ckpt. Use coverage as Stage 2. Remember to use a small lr.")
        loaded = torch.load(args.reload_ckpt)
        loaded['state_dict'] = strip_prefix_if_present(loaded['state_dict'], prefix='module.')
        model.load_state_dict(loaded['state_dict'], strict=False)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.schedule_step, gamma=0.1)
    logger.info('[4] Start training......')
    max_score = {'rouge_1_f_score': 0, 'rouge_2_f_score': 0, 'rouge_l_f_score': 0}
    best_epoch = 0
    args.ckpt = os.path.join(args.save_dir, 'best_model.pt')
    for epoch_num in range(args.max_epoch):
        model.train()
        if train_loader.example_num > 0:
            try:
                for batch_iter, train_batch in tqdm(enumerate(train_loader.gen_batch()), total=train_loader.example_num):
                    if train_batch[0] == None:
                        # empty input
                        continue

                    progress = epoch_num + batch_iter / train_loader.example_num

                    input_ids, attention_mask, labels = [a.to(device) for a in train_batch]

                    loss = model(input_ids, attention_mask, labels)

                    optimizer.zero_grad()

                    loss.backward()

                    if args.clip_value > 0:
                        nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
                    if args.clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                    optimizer.step()

                    meters.update(loss=loss)
                    if train_loader.example_num // 100 != 0:
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
                            if args.debug:
                                logger.info("[debug] start valide at first")
                                break
            except RuntimeError:
                # In python 3.7, if batch iterator raise StopIteration, it will be like a RuntimeError
                logger.info(f"stop at {batch_iter} on {epoch_num} epoch")

        score = validate(valid_loader, model, args.beam_size, args.max_dec_len, args.min_dec_len, device, args.save_dir,
                         fast=True)
        logger.info("val")
        logger.info(score)
        save = {
            'kwargs': model_kwargs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step()
        if avg(score) > avg(max_score):
            best_epoch = epoch_num
            max_score = score
            torch.save(save, args.ckpt)
            # os.path.join(args.save_dir, 'model_epoch%d_val%.3f.pt' % (epoch_num, score['rouge_1_f_score'])))

    logger.info("Best epoch is %d" % best_epoch)
    logger.info(max_score)
    logger.info('[5] Evaluate best epoch')
    args.test = True
    args.fast = False
    args.raw = False
    call_validate(args)


def parse_args():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--data_path',
                        default='/data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/topic1',
                        help='pickle file obtained by dataset dump or datadir for torchtext')

    parser.add_argument('--save_dir',
                        default='/data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/topic0/bart_ft',
                        help='path to save checkpoints and logs')
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--max-tgt-len', type=int, default=256)
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
    # parser.add_argument('--model-name', type=str, default='facebook/bart-large')
    parser.add_argument('--model-name', type=str, default='facebook/bart-base')

    parser.add_argument('--debug', action='store_true')

    # decode parameters
    parser.add_argument('--min_dec_len', type=int, default=10)
    parser.add_argument('--max_dec_len', type=int, default=200)
    parser.add_argument('--max_dec_sent', type=int, default=15)
    parser.add_argument('--beam_size', type=int, default=4)

    # data parameters
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--train_data_num', type=int, default=-1,
                        help='# of training set that can be used, -1 means all')

    args = parser.parse_args()

    model_clean_name = args.model_name.split("/")[-1]
    print(f"model_clean_name is {model_clean_name}")
    args.save_dir = os.path.join(args.data_path, f'{model_clean_name}_fine_tune')
    if args.train_data_num == -1:
        args.save_dir = os.path.join(args.save_dir, f'all_ml{args.max_len}_mtl{args.max_tgt_len}_me{args.max_epoch}')
    else:
        args.save_dir = os.path.join(args.save_dir,
                                     f'few_shot{args.train_data_num}_ml{args.max_len}_mtl{args.max_tgt_len}_me{args.max_epoch}')

    if args.reload_ckpt:
        args.save_dir = os.path.join(args.save_dir, 'reload')

        # override model-related arguments when reloading
        model_arg_names = {'max_dec_len', 'beam_size', 'min_dec_len'}
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
        # args.save_dir = os.path.join(args.save_dir, 'new')
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    logger = setup_logger("Bart", args.save_dir)

    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    train(args, logger)


if __name__ == '__main__':
    main()
