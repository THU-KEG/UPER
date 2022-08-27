import os
import random
import argparse
import json

seed_dict = {
    "company": [2198, 1446, 312, 1915, 1766, 2020, 743, 714, 1980, 890, 1312, 2735, 2322, 2154, 280],
    "film": [503, 2494, 2574, 1298, 1067, 26, 2438, 745, 2708, 1853, 925, 1753, 939, 1320, 1708],
    "animal": [1579, 715, 1133, 1198, 813, 1972, 619, 257, 1961, 1463, 1485, 1521, 63, 616, 35]
}

result_paths = {
    'company': {
        'wikicatsum': '/data/zfw/wikicatsum/company/output',
        'twag': '/data1/tsq/WikiGen/generate_models/30_titles/company/test_id_model_epoch8_val0.339/extra_sum',
        'twag-unk': '/data1/tsq/WikiGen/generate_models/30_titles/company/test_id_model_epoch8_val0.339/sum',
        # 'twag-hard': '/data/zfw/WikiGen/generate_models/company/test_id_model_epoch4_val0.264/sum',
        'bart_old': '/data1/tsq/WikiGen/bart_base/company_200/test_bart_data_model_epoch4_val0.305/ignored/sum',
        'bart_base_origin': '/data1/tsq/WikiGen/bart_base/company_200/test_bart_data_model_epoch4_val0.305/ignored/sum',
        'bart_base_max': '/data1/tsq/contrastive/group/animal/bart_whole_combined/oracle/test/',
        'bart_base_random': '/data1/tsq/contrastive/group/animal/bart_whole_combined/random/test/',
        'bart_base_min': '/data1/tsq/contrastive/group/animal/bart_whole_combined/oracle/test/',
        # 'hiersumm': '/data1/tsq/WikiGen/hiersumm/company/ckpt/sum',
        # 'tf-s2s': '/data1/tsq/wikisum/company/decoder/ignored/sum',
        'golden': '/data1/tsq/WikiGen/bart_base/company_200/test_bart_data_model_epoch4_val0.305/ignored/ref'
    },
    'film': {
        'wikicatsum': '/data/zfw/wikicatsum/film/output',
        'twag': '/data1/tsq/WikiGen/generate_models/3e-5/film/test_id_model_epoch9_val0.396/extra_sum',
        'twag-unk': '/data1/tsq/WikiGen/generate_models/3e-5/film/test_id_model_epoch9_val0.396/sum',
        'twag-hard': '/data1/tsq/WikiGen/generate_models/c_generate/film/test_id_model_epoch5_val0.348/sum',
        'bart': '/data1/tsq/WikiGen/bart_base/film_200/test_bart_data_model_epoch4_val0.380/ignored/sum',
        'hiersumm': '/data1/tsq/WikiGen/hiersumm/film/ckpt/sum',
        'tf-s2s': '/data1/tsq/wikisum/film/decoder/ignored/sum',
        'golden': '/data1/tsq/WikiGen/bart_base/film_200/test_bart_data_model_epoch4_val0.380/ignored/ref'
    },
    'animal': {
        # 'wikicatsum': '/data/zfw/wikicatsum/animal/output',
        # 'twag': '/data1/tsq/WikiGen/generate_models/20_titles/5_topics/animal/test_id_model_epoch9_val0.441/',
        # 'twag-unk': '/data1/tsq/WikiGen/generate_models/20_titles/5_topics/animal/test_id_model_epoch9_val0.441/',
        # 'bart_base_origin': '/data1/tsq/contrastive/rerank_documents/animal/bart_no_rerank/bart-base_fine_tune/test_bart_no_rerank_model_epoch4_val0.490',
        # 'bart_base_max': '/data1/tsq/contrastive/group/animal/bart_whole_combined/oracle/test/',
        # 'bart_base_random': '/data1/tsq/contrastive/group/animal/bart_whole_combined/random/test/',
        # 'bart_base_min': '/data1/tsq/contrastive/group/animal/bart_whole_combined/oracle/test/',
        # 'groupsum_c1_origin': '/data1/tsq/contrastive/group/animal/checkpoints/1centers/test_text_model_epoch7_val0.435',
        # 'groupsum_c2_origin': '/data1/tsq/contrastive/group/animal/checkpoints/2centers/test_text_model_epoch7_val0.432',
        'groupsum_c3_origin': '/data1/tsq/contrastive/group/animal/checkpoints/3centers/test_text_model_epoch4_val0.409',
        # 'groupsum_c4_origin': '/data1/tsq/contrastive/group/animal/checkpoints/4centers/test_text_model_epoch5_val0.396',
        # 'groupsum_c5_origin': '/data1/tsq/contrastive/group/animal/checkpoints/history/5centers_test_epoch5',
        'groupsum_c3_max': '/data1/tsq/contrastive/group/animal/clust3_combined/oracle/test/',
        'groupsum_c3_random': '/data1/tsq/contrastive/group/animal/clust3_combined/random/test/',
        'groupsum_c3_min': '/data1/tsq/contrastive/group/animal/clust3_combined/min/test/',
        'golden': '/data1/tsq/WikiGen/bart_base/animal/test_bart_data_model_epoch4_val0.392/ignored/ref'
    },
    'multi-news':{
        "c3":"/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers/test_text_model_epoch7",
        "golden": "/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers/test_text_model_epoch7/ref"
    }
}

wcs_files = []

data_paths = {
    'company': ('/data/zfw/wikicatsum/company/test.src', '/data/zfw/wikicatsum/company/test_ignoredIndices.log'),
    'film': ('/data/zfw/wikicatsum/film/test.src', '/data/zfw/wikicatsum/film/test_ignoredIndices.log'),
    'animal': ('/data/zfw/wikicatsum/animal/test.src', '/data/zfw/wikicatsum/animal/test_ignoredIndices.log'),
}


def gen_index_tuples(cate):
    total_examples = 0
    with open(data_paths[cate][0], 'r') as fin:
        total_examples = len(fin.readlines())

    ignore_index = set()
    with open(data_paths[cate][1], 'r') as fin:
        for line in fin:
            ignore_index.add(int(line.strip()))

    index_mapping = {}
    avail_index = 0
    for i in range(total_examples):
        if not (i in ignore_index):
            index_mapping[i] = avail_index
            avail_index += 1
    global wcs_files
    _f = os.listdir(result_paths[cate]['wikicatsum'])
    index_tuples = []
    for x in _f:
        if (x[-4:] != '.dec'):
            continue
        wcs_index = int(x[:-4])
        index_tuples.append((wcs_index, index_mapping[wcs_index]))

    index_tuples.sort(key=lambda x: x[0])

    return index_tuples, index_mapping


def sample_normal(path, index):
    with open(os.path.join(path, str(index[1]) + '_decoded.txt'), 'r') as fin:
        res = fin.read()

    return res


def sample_golden(path, index):
    with open(os.path.join(path, str(index[1]) + '_reference.txt'), 'r') as fin:
        res = fin.read()

    return res


def sample_candidate(path, index, beam_size=16):
    res = []
    with open(os.path.join(path, str(index[1]) + '_candidates.txt'), 'r') as fin:
        res_lines = fin.readlines()
        for i, res_line in enumerate(res_lines):
            if i % beam_size == 0:
                res.append(res_line)

    return res

def sample_doc(path, index, beam_size=16):
    res = []
    with open(os.path.join(path, str(index[1]) + '_documents.txt'), 'r') as fin:
        res_lines = fin.readlines()
    for i, res_line in enumerate(res_lines):
        # if i % beam_size == 0:
        res.append(res_line)

    return res

def sample_wcs(path, index):
    with open(os.path.join(path, str(index[0]) + '.dec'), 'r') as fin:
        res = fin.read()

    return res


read_functions = {
    'wikicatsum': sample_wcs,
    'twag': sample_normal,
    'twag-unk': sample_normal,
    'twag-hard': sample_normal,
    'bart': sample_normal,
    'hiersumm': sample_normal,
    'tf-s2s': sample_normal,
    'golden': sample_golden,
}


def parse_args():
    parser = argparse.ArgumentParser()
    # parameters

    parser.add_argument('--seed', type=int, default=2198, help='random seed')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])

    parser.add_argument('--no_random', action='store_true',
                        help='use random or not')

    parser.add_argument('--print_sum', action='store_true',
                        help='print sum or candidate')

    parser.add_argument('--extract_txt', action='store_true',
                        help='extract one category txt by seeds')

    parser.add_argument('--watch_doc', action='store_true',
                        help='watch the docs')

    parser.add_argument('--check_ref', action='store_true',
                        help='check whether reference is same')

    parser.add_argument('--save_dir', default='/home/tsq/WikiGen/src/evaluate', help='dir to save 15 passage')

    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.category)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    return args


def get_index(seed, index_mapping, index_tuples):
    for index, avail_index in index_mapping.items():
        if seed == index:
            return index_tuples.index((index, avail_index))


def main():
    args = parse_args()
    cate = args.category
    # index_tuples, index_mapping = gen_index_tuples(cate)
    index_mapping = None
    index_tuples = [(i, i) for i in range(2000)]
    num_examples = len(index_tuples)

    if args.extract_txt:
        seeds = seed_dict[cate]
        for seed in seeds:
            index = get_index(seed, index_mapping, index_tuples)
            res_dict = {}
            for mod in result_paths[cate]:
                res_dict[mod] = read_functions[mod](result_paths[cate][mod], index_tuples[index])
            # output
            json_file_path = os.path.join(args.save_dir, '{}.json'.format(seed))
            fout = open(json_file_path, "w")
            # print(res_dict)
            json.dump(res_dict, fout)
            fout.close()
        quit()

    if args.no_random:
        index = get_index(args.seed, index_mapping, index_tuples)
    else:
        index = random.randint(0, num_examples - 1)

    print(index_tuples[index])
    if args.print_sum:
        for mod in result_paths[cate]:
            print("\n" + mod + ":\n")

            if mod == "twag":
                sum_dir = os.path.join(result_paths[cate][mod], "extra_sum")
            elif mod == "golden":
                sum_dir = result_paths[cate][mod]
            else:
                sum_dir = os.path.join(result_paths[cate][mod], "sum")
            try:
                print(read_functions[mod](sum_dir, index_tuples[index]))
            except KeyError:
                print(sample_normal(sum_dir, index_tuples[index]))

    elif args.watch_doc:
        doc_dir = os.path.join(
            # "/data1/tsq/contrastive/group/animal/checkpoints/fixed/3centers/test_text_model_epoch9/doc/"
            "/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers/test_text_model_epoch7/doc"
        )
        docs = sample_doc(doc_dir, index_tuples[index])
        print(f"doc_num is {len(docs)}")
        for doc in docs:
            print(doc)

    else:
        candidate_dir = os.path.join(
            # "/data1/tsq/contrastive/group/animal/checkpoints/3centers/test_text_model_epoch4_val0.409/",
            "/data1/tsq/contrastive/group/multi_news/checkpoints/fixed/3centers/test_text_model_epoch7",
            "candidate")
        candidates = sample_candidate(candidate_dir, index_tuples[index])
        print(f"candidate_num is {len(candidates)}")
        for candidate in candidates:
            print(candidate)

    if args.check_ref:
        # check whether reference is same
        print("#" * 16)
        print("check whether reference is same")
        print("#" * 16)
        for mod in result_paths[cate]:
            print(index_tuples[index])
            print(mod)
            if mod == "golden":
                ref_dir = result_paths[cate][mod]
            elif mod == "twag":
                ref_dir = os.path.join(result_paths[cate][mod], "extra_ref")
            else:
                ref_dir = os.path.join(result_paths[cate][mod], "ref")
            print(sample_golden(ref_dir, index_tuples[index]))


if __name__ == "__main__":
    main()
