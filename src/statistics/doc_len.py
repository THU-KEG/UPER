import argparse
import os
import json
import logging


def count_doc_length_wikicatsum(args):
    data_dir = os.path.join(args.data_dir, 'raw')
    splits = ['train', 'valid', 'test']

    total_data_num = 0
    total_token_num_src = 0
    total_sent_src = 0
    for split in splits:
        src_dir = os.path.join(data_dir, f"{split}_src")
        data_num = len(os.listdir(src_dir))
        total_data_num += data_num
        for data_id in range(data_num):
            src_path = os.path.join(src_dir, f"{data_id}.txt")
            with open(src_path, 'r') as fin:
                lines = fin.readlines()
                total_sent_src += len(lines)
                for line in lines:
                    total_token_num_src += len(line.strip().split())

    result = {
        'total_data_num': total_data_num,
        'avg_token_num_src': total_token_num_src / total_data_num,
        'avg_sent_num_src': total_sent_src / total_data_num
    }
    result['avg_sent_len'] = result['avg_token_num_src'] / result['avg_sent_num_src']

    with open(os.path.join(data_dir, 'doc_length.json'), 'w') as f:
        f.write(json.dumps(result, indent=4))
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    datasets = {
        'wikicatsum': count_doc_length_wikicatsum,
        # 'multi_news': count_length_multi_news,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='wikicatsum', choices=datasets.keys())
    parser.add_argument('--data-dir', type=str,
                        default='/data1/tsq/contrastive/clust_documents/animal'
                        # default='/data1/tsq/contrastive/group/multi_news/text.pkl'
                        )
    args = parser.parse_args()
    datasets[args.data](args)
