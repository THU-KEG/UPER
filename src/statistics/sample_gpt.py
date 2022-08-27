import os
import random
import argparse
import json
import random

ext_result_paths = {
    'animal': {
        # 'ml500_qa_p10': '/data1/tsq/contrastive/clust_documents/animal/result/tn4_lsn10/test_src/qa_para_penal10.0/level_all/mrl20_mtn500_sto_rem',
        # 'ml500_tf_idf': '/data1/tsq/contrastive/clust_documents/animal/result/tf_idf/mtn500',
        # 'ml500_inverse': '/data1/tsq/contrastive/clust_documents/animal/result/inverse_add4/test_src/inverse_score/mrl20_mtn500_sto_rem',
        # 'ml500_qa_la': '/data1/tsq/contrastive/clust_documents/animal/result/tn4_lsn10/test_src/qa_score/level_all/mrl20_mtn500_sto_rem',
        'ws0.75': '/data1/tsq/contrastive/clust_documents/animal/result/inverse_add4/test_src/inverse_score/ws_0.75/mrl64_mtn1000_no',
        'ws0.75_topic_avg': '/data1/tsq/contrastive/clust_documents/animal/result/inverse_add4/test_src/inverse_score/ws_0.75/zs_classify/k4_i64_o64/mrl64_mtn1000_no',
        'ws0.75_topic_max': '/data1/tsq/contrastive/clust_documents/animal/result/inverse_add4/test_src/inverse_score/ws_0.75/zs_classify/k4_i1024_o64_tp_max/mrl64_mtn1000_no',
    },
    'film': {
        'ws0.75': '/data1/tsq/contrastive/clust_documents/film/result/inverse_add0/test_src/inverse_score/ws_0.75/mrl64_mtn1000_no',
        'ws0.75_topic': '/data1/tsq/contrastive/clust_documents/film/result/inverse_add0/test_src/inverse_score/ws_0.75/zs_classify/k4_i1024_o64_tp_max/mrl64_mtn1000_no',
    },
    'company': {
        'ws0.75': '/data1/tsq/contrastive/clust_documents/company/result/inverse_add0/test_src/inverse_score/ws_0.75/mrl64_mtn1000_no',
        'ws0.75_topic': '/data1/tsq/contrastive/clust_documents/company/result/inverse_add0/test_src/inverse_score/ws_0.75/zs_classify/k4_i1024_o64_tp_max/mrl64_mtn1000_no',
    }
}

abs_result_paths = {
    'animal': {
        # 'wikicatsum': '/data/zfw/wikicatsum/animal/output',
        # 'twag': '/data1/tsq/WikiGen/generate_models/20_titles/5_topics/animal/test_id_model_epoch9_val0.441/',
        # 'unilm_tf_idf': '/data1/tsq/unilm/result/animal/tf_idf/ckpt-400000/test_result/mbn1_dr0.7_spl_fix_rem/',
        # 'unilm_ws0.75': '/data1/tsq/unilm/result/animal/power/ckpt-400000/test_result/mbn1_dr0.7_spl_fix_rem/',
        'bart_tf_idf': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tf_idf/bart-base_fine_tune/few_shot46773_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        'bart_ws0.75': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/inverse_add4/ws_0.75/inverse_add_title/bart-base_fine_tune/few_shot46773_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        # 'bart_ws0.75_topic_avg': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/inverse_add4/ws_0.75/zs_classify/k4_i64_o64/inverse_add_title/bart-base_fine_tune/few_shot46773_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        # 'bart_ws0.75_topic_max': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/inverse_add4/ws_0.75/zs_classify/k4_i1024_o64_tp_max/inverse_add_title/bart-base_fine_tune/few_shot46773_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        # 'twag-unk': '/data1/tsq/WikiGen/generate_models/20_titles/5_topics/animal/test_id_model_epoch9_val0.441/',
        # 'bart_base_origin': '/data1/tsq/contrastive/rerank_documents/animal/bart_no_rerank/bart-base_fine_tune/test_bart_no_rerank_model_epoch4_val0.490',
        # 'bart_base_max': '/data1/tsq/contrastive/group/animal/bart_whole_combined/oracle/test/',
        # 'bart_base_random': '/data1/tsq/contrastive/group/animal/bart_whole_combined/random/test/',
        # 'bart_base_min': '/data1/tsq/contrastive/group/animal/bart_whole_combined/oracle/test/',
        # 'fs500_qa_p10_bartl': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tn4_lsn10/qa_para_penal10.0/bart-large_fine_tune/few_shot500/test_qa_para_penal10_model_epoch1_val0.329',
        # 'fs500_tf_idf_bartl': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tf_idf/bart-large_fine_tune/few_shot500/test_tf_idf_model_epoch4_val0.306',
        # 'fs500_tf_idf_bartb': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tf_idf/bart-base_fine_tune/few_shot500/test_tf_idf_model_epoch4_val0.316',
        # 'fs500_inverse_bartl': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/inverse_add4/inverse_score/bart-large_fine_tune/few_shot500/test_inverse_score_model_epoch2_val0.351',
        # 'fs500_inverse_bartb': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/inverse_add4/inverse_score/bart-base_fine_tune/few_shot500/test_inverse_score_model_epoch3_val0.362',
        # 'fs500_qa_bartl': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tn4_lsn10/qa_score/bart-large_fine_tune/few_shot500/test_qa_score_model_epoch3_val0.293',
        # 'fs500_qa_bartb': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tn4_lsn10/qa_score/bart-base_fine_tune/few_shot500/test_qa_score_model_epoch2_val0.366',
        # 'fs200_qa_bartl': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tn4_lsn10/qa_score/bart-large_fine_tune/few_shot200/test_qa_score_model_epoch4_val0.341',
        # 'fs500_inverse_bartb': '',
        # 'golden': '/data1/tsq/WikiGen/bart_base/animal/test_bart_data_model_epoch4_val0.392/ignored/ref'
        'golden': '/data1/tsq/contrastive/clust_documents/animal/bart/test_as_valid/tf_idf/bart-base_fine_tune/few_shot46773_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem/ref'
    },
    'film': {
        'bart_tf_idf': '/data1/tsq/contrastive/clust_documents/film/bart/test_as_valid/tf_idf/bart-base_fine_tune/few_shot51399_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        'bart_ws0.75': '/data1/tsq/contrastive/clust_documents/film/bart/test_as_valid/inverse_add0/ws_0.75/inverse_add_title/bart-base_fine_tune/few_shot51399_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        # 'bart_ws0.75_topic': '/data1/tsq/contrastive/clust_documents/film/bart/test_as_valid/inverse_add0/ws_0.75/zs_classify/k4_i52_o52/inverse_add_title/bart-base_fine_tune/few_shot51399_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        'golden': '/data1/tsq/contrastive/clust_documents/film/bart/test_as_valid/tf_idf/bart-base_fine_tune/few_shot51399_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem/ref',

    },
    'company': {
        'bart_tf_idf': '/data1/tsq/contrastive/clust_documents/company/bart/test_as_valid/tf_idf/bart-base_fine_tune/few_shot52506_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        'bart_ws0.75': '/data1/tsq/contrastive/clust_documents/company/bart/test_as_valid/inverse_add0/ws_0.75/inverse_add_title/bart-base_fine_tune/few_shot52506_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        # 'bart_ws0.75_topic': '/data1/tsq/contrastive/clust_documents/company/bart/test_as_valid/inverse_add0/ws_0.75/zs_classify/k4_i48_o48/inverse_add_title/bart-base_fine_tune/few_shot52506_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem',
        'golden': '/data1/tsq/contrastive/clust_documents/company/bart/test_as_valid/tf_idf/bart-base_fine_tune/few_shot52506_ml1000_mtl256_me16/test_result/mbn1_dr0.7_spl_fix_rem/ref',
    },
    'multi-news': {
        "c3": "/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers/test_text_model_epoch7",
        "golden": "/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers/test_text_model_epoch7/ref"
    }
}


def sample_normal(path, index):
    with open(os.path.join(path, str(index) + '_decoded.txt'), 'r') as fin:
        res = fin.read()

    return res


def sample_golden(path, index):
    with open(os.path.join(path, str(index) + '_reference.txt'), 'r') as fin:
        res = fin.read()

    return res


def parse_args():
    parser = argparse.ArgumentParser()
    # parameters

    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--task', default='ext_abs', choices=['ext', 'ext_abs', 'extract_txt'])
    parser.add_argument('--save_dir', default='/home/tsq/TopCLS/human_evaluation')

    args = parser.parse_args()

    # args.save_dir = os.path.join(args.save_dir, args.category)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    return args


read_functions = {
    'twag': sample_normal,
    'twag-unk': sample_normal,
    'twag-hard': sample_normal,
    'bart': sample_normal,
    'hiersumm': sample_normal,
    'tf-s2s': sample_normal,
    'golden': sample_golden,
}


def read_from_result(result_paths, cate, index):
    for mod in result_paths[cate]:
        print("\n" + mod + ":\n")
        if mod == "twag":
            sum_dir = os.path.join(result_paths[cate][mod], "extra_sum")
        elif mod == "golden":
            sum_dir = result_paths[cate][mod]
        else:
            sum_dir = os.path.join(result_paths[cate][mod], "sum")
        try:
            print(read_functions[mod](sum_dir, index))
        except KeyError:
            print(sample_normal(sum_dir, index))

    print(f"index is: {index}")


def main():
    args = parse_args()
    cate = args.category
    if args.seed == -1:
        index = random.randint(0, 2000)
    else:
        index = args.seed

    if args.task == 'ext':
        print("extract result: ")
        read_from_result(ext_result_paths, cate, index)
    elif args.task == 'extract_txt':
        num = range(395, 1453)
        seeds = random.sample(num, 20)
        res = []
        for seed in seeds:
            index = seed
            text_dict = {}
            for mod in abs_result_paths[cate]:
                if mod == "twag":
                    sum_dir = os.path.join(abs_result_paths[cate][mod], "extra_sum")
                elif mod == "golden":
                    sum_dir = abs_result_paths[cate][mod]
                else:
                    sum_dir = os.path.join(abs_result_paths[cate][mod], "sum")
                try:
                    text_dict[mod] = read_functions[mod](sum_dir, index)
                except KeyError:
                    text_dict[mod] = sample_normal(sum_dir, index)
            res.append(text_dict)
        # output
        json_file_path = os.path.join(args.save_dir, '{}.json'.format(cate))
        fout = open(json_file_path, "w")
        # print(res_dict)
        json.dump(res, fout)
        fout.close()
        quit()
    else:
        # ext-abs
        print("extract result: ")
        read_from_result(ext_result_paths, cate, index)
        print("abstract result: ")
        read_from_result(abs_result_paths, cate, index)


if __name__ == "__main__":
    main()
