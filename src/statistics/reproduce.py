import argparse
import os
import json
from shutil import rmtree, copyfile
# from src.GroupSum.pyrouge.rouge import Rouge155
from src.bart.pyrouge.rouge import Rouge155

USE_LOG_DIR = True


def call_rouge(log_dir, sum_dir_name="sum", ref_dir_name="ref"):
    ref_dir = os.path.join(log_dir, ref_dir_name)
    sum_dir = os.path.join(log_dir, sum_dir_name)

    assert len(os.listdir(ref_dir)) == len(os.listdir(sum_dir))

    Rouge155_obj = Rouge155(stem=True, tmp=os.path.join(log_dir, 'tmp'))
    score = Rouge155_obj.evaluate_folder(sum_dir, ref_dir)

    with open(os.path.join(log_dir, 'scores.txt'), 'w') as f:
        f.write(json.dumps(score, indent=4))

    print(score)
    return score


def make_dir_rm(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        rmtree(dir)


def work():
    if USE_LOG_DIR:
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default="/data1/tsq/multi_news/result",
                            help='the root directory of txt')
        parser.add_argument('--baseline_name', type=str, required=True, help='the name of baseline')
        parser.add_argument('--ref_dir_name', type=str, default="ref", help='the name of ref directory ')
        parser.add_argument('--sum_dir_name', type=str, default="sum", help='the name of sum directory ')
        args = parser.parse_args()
        # get baseline summary
        baseline_txt_path = os.path.join(args.data_dir, f'{args.baseline_name}.txt')
        log_dir = os.path.join(args.data_dir, args.baseline_name)
        make_dir_rm(log_dir)
        sum_dir = os.path.join(log_dir, args.sum_dir_name)
        ref_dir = os.path.join(log_dir, args.ref_dir_name)
        make_dir_rm(sum_dir)
        make_dir_rm(ref_dir)
        # output sum
        with open(baseline_txt_path, 'r') as fin:
            lines = fin.readlines()
            for i, line in enumerate(lines):
                sum_path = os.path.join(sum_dir, f"{i}_decoded.txt")
                with open(sum_path, 'w') as fout:
                    fout.write(line)
        # output ref
        ref_origin_dir = '/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers/test_text_model_epoch7/ref'
        data_num = len(os.listdir(ref_origin_dir))
        assert len(os.listdir(sum_dir)) == data_num
        for i in range(data_num):
            old_ref = os.path.join(ref_origin_dir, f"{i}_reference.txt")
            new_ref = os.path.join(ref_dir, f"{i}_reference.txt")
            copyfile(old_ref, new_ref)

        call_rouge(log_dir, args.sum_dir_name, args.ref_dir_name)
        return


if __name__ == '__main__':
    work()
