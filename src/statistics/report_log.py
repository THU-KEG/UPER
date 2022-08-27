import argparse
import os
import re
import json

val_pattern = re.compile("val on at epoch(\d)")
val_center_pattern = re.compile("val on (\d)centers at epoch(\d)")
rouge_pattern = re.compile("\{.*\}")


def get_need_dict(rouge_dict, keys):
    need_dict = {}
    need_dict["epoch"] = rouge_dict["epoch"]
    for key in keys:
        need_dict[key] = rouge_dict[key]
    # Besides the keys, we also report avg
    avg_f_12l = (rouge_dict['rouge_1_f_score'] + rouge_dict['rouge_2_f_score'] + rouge_dict['rouge_l_f_score']) / 3
    avg_f_12su4 = (rouge_dict['rouge_1_f_score'] + rouge_dict['rouge_2_f_score'] + rouge_dict['rouge_su4_f_score']) / 3
    need_dict['avg_f_12l'] = avg_f_12l
    need_dict['avg_f_12su4'] = avg_f_12su4
    return need_dict


def report_scores(log_path, center_decision, keys, mode, sort_by):
    # rouge score for each epoch
    rouge_dicts = []
    center_num = 0
    epoch = 0
    with open(log_path, "r") as fin:
        is_rouge = False
        lines = fin.readlines()
        for line in lines:
            # Search whether the line is val
            if center_decision == 'fixed':
                valObj = val_pattern.search(line)
                if valObj:
                    is_rouge = True
                    epoch = valObj.group(1)
                    continue
            elif center_decision == 'dynamic_default':
                # there will be many centers in one epoch
                val_center = val_center_pattern.search(line)
                if val_center:
                    is_rouge = True
                    center_num = val_center.group(1)
                    epoch = val_center.group(2)
                    continue
            # This line is the json dict of rouge score
            if is_rouge:
                is_rouge = False
                rougeObj = rouge_pattern.search(line)
                if rougeObj:
                    json_str = rougeObj.group()
                    new_str = json_str.replace("'", "\"")
                    rouge_dict = json.loads(new_str)
                    rouge_dict["epoch"] = epoch
                    if center_decision == 'dynamic_default':
                        rouge_dict["center_num"] = center_num
                    rouge_dicts.append(rouge_dict)
    # report each epoch
    need_dicts = []
    for rouge_dict in rouge_dicts:
        need_dict = get_need_dict(rouge_dict, keys)
        if mode == 'epoch':
            print(f"Epoch {rouge_dict['epoch']}")
            print(need_dict)
        need_dicts.append(need_dict)
    # sort and report max
    sorted_need_dicts = sorted(need_dicts, key=lambda x: x[sort_by], reverse=True)
    best_dict = sorted_need_dicts[0]

    if center_decision == 'fixed':
        print(f"The epoch with max {sort_by} score is epoch{best_dict['epoch']}, its score is:")
    else:
        print(f"The epoch with max {sort_by} score is epoch{best_dict['epoch']} with {best_dict['center_num']}centers.")
        print("Its score is:")
    print(best_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        # default='/data1/tsq/contrastive/group/animal/checkpoints/fixed/3centers',
                        # default='/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/5centers',
                        # default='/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers',
                        default='/data1/tsq/contrastive/group/multi_news/checkpoints/dynamic_default',
                        help="where the log.txt store")
    parser.add_argument('--center_decision', default='dynamic_default',
                        choices=['fixed', 'dynamic_default', 'dynamic_predict'],
                        help="how to decide the center")
    parser.add_argument('--mode', default='max', choices=['max', 'epoch'],
                        help="if mode is set max, we only report the epoch that has max rouge performance")
    parser.add_argument('--sort_by', default='avg_f_12su4',
                        choices=['avg_f_12l', 'avg_f_12su4', 'rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score',
                                 'rouge_su4_f_score'])
    parser.add_argument("--keys", nargs='+', type=str,
                        default=['rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score', 'rouge_su4_f_score'])

    args = parser.parse_args()
    data_path = os.path.join(args.data_dir, "log.txt")
    if args.center_decision == 'dynamic_default':
        args.keys.append('center_num')
    report_scores(data_path, args.center_decision, args.keys, args.mode, args.sort_by)
