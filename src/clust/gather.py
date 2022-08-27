import argparse
import random
import pickle
from tqdm import tqdm
import os
import json
from src.bart.preprocess import check_fout
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cat2train_topic = {
    'animal': {'hard_topic': [0.41220223435067876, 0.17722597678954938, 0.23505999645341163, 0.17551179240636022],
               'soft_topic': [11.029251379821766, 9.867149175193736, 10.36742684244482, 10.143616246324191]},
    'company': {'hard_topic': [0.0033407389122027484, 0.04610384267259113, 0.31666255245618363, 0.6338928659590225],
                'soft_topic': [3.890508188710659, 6.532136608188069, 8.620944056495206, 9.418548873625802]},
    'film': {'hard_topic': [0.03876355843888979, 0.23751791891111732, 0.0454859021590759, 0.678232620490917],
             'soft_topic': [4.93062380184039, 5.824358654320547, 5.2763890363700465, 7.663819638424197]},
    'wcep': {}
    # film ln20
    # {'hard_topic': [0.0035568732784390855, 0.0022163185627687835, 0.1201606708644621, 0.87406613729433],
    #      'soft_topic': [4.355642850557822, 1.8762615868672385, 7.74986236194491, 9.786901120792738]}
}


def make_dirs(dir):
    if not (os.path.exists(dir)):
        os.makedirs(dir)


def write_gold_score_json(top_k_src, bottom_k_src, json_path):
    json_fout = check_fout(json_path)
    for src in top_k_src:
        src['salient'] = True
        json_fout.write(json.dumps(src))
        json_fout.write('\n')

    for src in bottom_k_src:
        src['salient'] = False
        json_fout.write(json.dumps(src))
        json_fout.write('\n')


def get_pattern_num(pattern_dir):
    test_dir = os.path.join(pattern_dir, 'test')
    test_data_path = os.path.join(test_dir, "0_pattern.json")
    pattern_num = 0
    with open(test_data_path, 'r') as json_fin:
        line = json_fin.read()
        pattern_clust = json.loads(line)
        for pattern_list in pattern_clust:
            pattern_num += len(pattern_list)

    return pattern_num, pattern_clust


def normalize_score_dicts(score_dicts, ws, tf_idf_lines, prompt):
    # for ppl based score, the lower score is, the higher quality this sentence has.
    # however, for tf-idf and rouge, the higher tf_idf is , the higher quality the sentence has.
    is_ppl = prompt in ['none', 'inverse', 'qa']
    scaler = StandardScaler()
    scores = [[score_dict['score']] for score_dict in score_dicts]
    normalized_scores = scaler.fit_transform(scores)
    for i, normalized_score in enumerate(normalized_scores):
        tf_idf = float(tf_idf_lines[i].strip())
        if is_ppl:
            score_dicts[i]['score'] = normalized_score[0] * ws - (1 - ws) * tf_idf
        else:
            # score is rouge
            score_dicts[i]['score'] = normalized_score[0] * ws + (1 - ws) * tf_idf
    return score_dicts


def get_sort_function(level, topic_score, scores_list, prompt):
    if prompt == 'qa':
        if level == 'all':
            return sum([sum(scores) for scores in scores_list])
        elif level == 'topic':
            if topic_score == 'min':
                return [min(scores) for scores in scores_list]
            else:
                # avg
                return [sum(scores) / len(scores) for scores in scores_list]

        else:
            # lead section
            return scores_list
    elif prompt == 'inverse':
        # inverse
        if len(scores_list) == 0:
            return 0
        return sum(scores_list) / len(scores_list)
    elif prompt == 'rouge':
        if level == 'recall':
            scores_list = [scores_list[0], scores_list[3], scores_list[6]]
        elif level == 'precision':
            scores_list = [scores_list[1], scores_list[4], scores_list[7]]
        elif level == 'f1':  # f1
            scores_list = [scores_list[2], scores_list[5], scores_list[8]]
        elif level == 'r1_recall':
            scores_list = [scores_list[0]]
        elif level == 'r2_recall':
            scores_list = [scores_list[3]]
        elif level == 'rl_recall':
            scores_list = [scores_list[6]]
        return sum(scores_list) / len(scores_list)
    else:
        # none prompt
        return scores_list[0]


def sort_ppl_score(args, sentences_dir, score_dir, pattern_dir, tgt_dir, split,
                   tf_policy, clust_policy, clust_num, clust_input_sent_num, clust_output_sent_num, proportion,
                   attenuation_coefficient, start_id, end_id, level,
                   topic_score, prompt, para_penal, rm_each_sent=False):
    # path
    if prompt == 'qa' or prompt == 'rouge':
        result_dir = os.path.join(tgt_dir, f"level_{level}")
        if level == 'topic':
            result_dir = os.path.join(result_dir, topic_score)
    else:
        result_dir = tgt_dir
    if tf_policy != 'no':
        result_dir = os.path.join(result_dir, tf_policy)
    if clust_policy != 'no':
        if clust_policy != 'k_means':
            clust_num = args.topic_num
        else:
            sent_embeddings_path = os.path.join(sentences_dir, f'{split}_embeddings.pkl')
            with open(sent_embeddings_path, 'rb') as fin:
                data_id2sent_embeddings = pickle.load(fin)
        if args.last_noisy:
            result_dir = os.path.join(result_dir, clust_policy,
                                      f'k{clust_num}_i{clust_input_sent_num}_o{clust_output_sent_num}_{proportion}_noise')
        elif args.max_logit:
            result_dir = os.path.join(result_dir, clust_policy,
                                      f'k{clust_num}_i{clust_input_sent_num}_o{clust_output_sent_num}_{proportion}_max')
        else:
            result_dir = os.path.join(result_dir, clust_policy,
                                      f'k{clust_num}_i{clust_input_sent_num}_o{clust_output_sent_num}_{proportion}')

        if proportion not in ['free', 'tp']:
            result_dir = os.path.join(result_dir, f"{attenuation_coefficient}")

    clust_params = {
        "clust_policy": clust_policy,
        "clust_num": clust_num,
        "clust_input_sent_num": clust_input_sent_num,
        "clust_output_sent_num": clust_output_sent_num,
        "last_noisy": args.last_noisy,
        "proportion": args.proportion,
        "distribution": cat2train_topic[args.category],
        "attenuation_coefficient": attenuation_coefficient,
    }
    make_dirs(result_dir)

    top_bottom_dir = os.path.join(result_dir, "top_bottom")
    make_dirs(top_bottom_dir)

    sorted_dir = os.path.join(result_dir, "sorted")
    make_dirs(sorted_dir)

    # read score
    if rm_each_sent:
        para2sents_json_path = os.path.join(sentences_dir, f"{split}_para2sents.json")
        with open(para2sents_json_path, 'r') as jin:
            para2sents_list = json.loads(jin.read())

    for data_id in tqdm(range(start_id, end_id)):
        sentences_path = os.path.join(sentences_dir, f"{split}_src", f"{data_id}.txt")
        bl_path = os.path.join(sentences_dir, f"{split}_bart_len", f"{data_id}.txt")
        if args.last_noisy:
            topic_path = os.path.join(args.classify_dir, split, f'ls{args.lead_section_num}_t{args.topic_num}_noise',
                                      f"{data_id}_class.json")
        elif args.max_logit:
            topic_path = os.path.join(args.classify_dir, split, f'ls{args.lead_section_num}_t{args.topic_num}',
                                      'max_logit', f"{data_id}_class.json")
        else:
            topic_path = os.path.join(args.classify_dir, split, f'ls{args.lead_section_num}_t{args.topic_num}',
                                      f"{data_id}_class.json")

        tf_path = os.path.join(sentences_dir, f"{split}_tf", f"{data_id}.txt")
        tf_idf_path = os.path.join(sentences_dir, f"{split}_tf_idf", f"{data_id}.txt")
        sent_lines = open(sentences_path, 'r').readlines()
        if clust_policy == 'no':
            bl_lines = None
            topic_lines = None
            tf_lines = None
        else:
            bl_lines = open(bl_path, 'r').readlines()
            topic_lines = open(topic_path, 'r').readlines()
            tf_lines = open(tf_path, 'r').readlines()
        tf_idf_lines = open(tf_idf_path, 'r').readlines()
        if not rm_each_sent:
            scores_path = os.path.join(score_dir, f"{data_id}_score.txt")
        else:
            scores_path = os.path.join(score_dir, f"{data_id}_score.json")
        score_lines = open(scores_path, 'r').readlines()
        if clust_policy == 'k_means':
            clust_params["embeddings"] = data_id2sent_embeddings[data_id]
        # get score for each sentence
        score_dicts = []
        if rm_each_sent:
            para2sents = para2sents_list[data_id]
            for para_id, sent_ids in enumerate(para2sents.values()):
                try:
                    sent_id2para_socre = json.loads(score_lines[int(para_id)].strip())
                except IndexError:
                    # when len(para2sents) is 119, in fact it has 120 keys?
                    # print(len(para2sents))
                    # print(len(score_lines))
                    continue
                origin_score_list = sent_id2para_socre['origin']
                para_origin_score = get_sort_function(level, topic_score, origin_score_list, prompt)
                for sent_id in sent_ids:
                    sent_score_dict = {'sent_id': sent_id, 'origin': para_origin_score}
                    #  the more para_origin_score is, the less important the paragraph is
                    if len(sent_ids) > 1:
                        rm_sent_score_list = sent_id2para_socre[str(sent_id)]
                        rm_sent_score = get_sort_function(level, topic_score, rm_sent_score_list, prompt)
                        #  the more ppl_gain is, the more important the sentence is
                        ppl_gain = rm_sent_score - para_origin_score
                        sent_score_dict['ppl_gain'] = ppl_gain
                        # lower score are higher ranked
                        sent_score_dict['score'] = para_origin_score * para_penal - ppl_gain
                    else:
                        # para has only one sentence
                        sent_score_dict['score'] = para_origin_score * para_penal

                    score_dicts.append(sent_score_dict)
        else:
            for sent_id, score_line in enumerate(score_lines):
                score_dict = {'sent_id': sent_id}
                if prompt == 'qa' or prompt == 'inverse' or prompt == 'rouge':
                    # 'qa' key: sent_id, value: scores_list [[float]] (i: topic_id j: lead_section_id)
                    # 'inverse' key: sent_id, value: scores_list [float] (i: pattern_id)
                    scores_list = json.loads(score_line.strip())
                    score_dict['scores_list'] = scores_list
                    score_dict['score'] = get_sort_function(level, topic_score, scores_list, prompt)

                else:
                    # prompt is none or regression
                    score_dict['score'] = float(score_line.strip())
                score_dicts.append(score_dict)

        if tf_policy[:2] == 'ws':
            ws = float(tf_policy.split('_')[-1])
            # normalize the score_dict['score'] and let score = ws * score - (1 - ws) * tf_idf if is_ppl
            try:
                score_dicts = normalize_score_dicts(score_dicts, ws, tf_idf_lines, prompt)
            except ValueError:
                print(f"{data_id} document has no tf_idf scores")
            except IndexError:
                print(f"{data_id} document has no sentences")

        # sort and output
        sort_and_output(prompt, clust_params, pattern_dir, level, score_dicts, sent_lines, bl_lines, tf_lines,
                        sorted_dir, top_bottom_dir, data_id, tf_policy, topic_lines)

    return result_dir


def sort_and_output(prompt, clust_params, pattern_dir, level, score_dicts, sent_lines, bl_lines, tf_lines, sorted_dir,
                    top_bottom_dir, data_id, tf_policy, topic_lines):
    isRouge = prompt not in ['none', 'qa', 'inverse']
    clust_params["isRouge"] = isRouge
    clust_params["topic_lines"] = topic_lines
    if prompt == 'qa':
        pattern_num, pattern_clust = get_pattern_num(pattern_dir)
        topic_num = len(pattern_clust)
        # print(f"Total pattern_num is {pattern_num}")
        if level == 'all':
            ground_scores = sorted(score_dicts, key=lambda x: x['score'], reverse=False)
            out_sorted(ground_scores, sent_lines, bl_lines, tf_lines, pattern_num, sorted_dir, top_bottom_dir, data_id,
                       tf_policy, clust_params)
        elif level == 'topic':
            for topic_id in range(topic_num):
                # each topic will sort and output once
                ground_scores = sorted(score_dicts, key=lambda x: x['score'][topic_id], reverse=False)
                out_sorted(ground_scores, sent_lines, bl_lines, tf_lines, pattern_num, sorted_dir, top_bottom_dir,
                           f"{data_id}_t{topic_id}", tf_policy, clust_params)
        else:
            # lead_section
            for topic_id, sections in enumerate(pattern_clust):
                for sec_id in range(len(sections)):
                    # each lead section will sort and output once
                    ground_scores = sorted(score_dicts, key=lambda x: x['score'][topic_id][sec_id], reverse=False)
                    out_sorted(ground_scores, sent_lines, bl_lines, tf_lines, pattern_num, sorted_dir, top_bottom_dir,
                               f"{data_id}_t{topic_id}_s{sec_id}", tf_policy, clust_params)
    else:
        # if is rouge, the larger rouge is, the higher it should rank
        ground_scores = sorted(score_dicts, key=lambda x: x['score'], reverse=isRouge)
        out_sorted(ground_scores, sent_lines, bl_lines, tf_lines, 5, sorted_dir, top_bottom_dir, data_id, tf_policy,
                   clust_params)


def get_sent_embeddings(ground_scores, embeddings):
    """
    :param ground_scores: [dict]
    :param embeddings: [tensor]
    :return: [tensor] embeddings in the order of ground_scores's sent_id
    """
    sent_embeddings = []
    for ground_score in ground_scores:
        sent_id = ground_score['sent_id']
        sent_embeddings.append(embeddings[sent_id])
    return sent_embeddings


def k_means(ground_scores, sent_lines, sorted_sents_fout, top_id_fout, clust_params):
    if len(ground_scores) < clust_params["clust_output_sent_num"]:
        # no need to perform clustering, output them all
        for ground_score in ground_scores:
            sent_id = ground_score['sent_id']
            sorted_sents_fout.write(sent_lines[sent_id])
    else:
        ground_scores = ground_scores[:clust_params["clust_input_sent_num"]]
        sent_embeddings = get_sent_embeddings(ground_scores, clust_params["embeddings"])
        # Perform k-means clustering
        num_clusters = clust_params["clust_num"]
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(sent_embeddings)
        cluster_assignment = clustering_model.labels_
        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(ground_scores[sentence_id])
        sort_and_output_each_clust(clustered_sentences, sent_lines, sorted_sents_fout, top_id_fout, num_clusters,
                                   clust_params["clust_output_sent_num"], clust_params["isRouge"])


def zs_classify(ground_scores, sent_lines, bl_lines, sorted_sents_fout, top_id_fout, clust_params):
    """
    if len(ground_scores) < clust_params["clust_output_sent_num"]:
        # no need to perform clustering, output them all
        for ground_score in ground_scores:
            sent_id = ground_score['sent_id']
            sorted_sents_fout.write(sent_lines[sent_id])
    else:
    """

    ground_scores = ground_scores[:clust_params["clust_input_sent_num"]]
    num_clusters = clust_params["clust_num"]
    topic_lines = clust_params["topic_lines"]
    cluster_assignment = []
    for doc_id, topic_line in enumerate(topic_lines):
        topic_dict = json.loads(topic_line.strip())
        assert doc_id == topic_dict["doc_id"]
        cluster_assignment.append(topic_dict["topic"])
    clustered_sentences = [[] for i in range(num_clusters)]
    noisy_docs = []
    for doc_id, ground_score in enumerate(ground_scores):
        sentence_id = ground_score['sent_id']
        cluster_id = cluster_assignment[sentence_id]
        if cluster_id >= num_clusters:
            # last noisy
            noisy_docs.append(ground_score)
            continue
        clustered_sentences[cluster_id].append(ground_score)
    not_noisy_num = sort_and_output_each_clust(ground_scores, clustered_sentences, sent_lines, bl_lines,
                                               sorted_sents_fout, top_id_fout, clust_params["attenuation_coefficient"],
                                               num_clusters, clust_params["proportion"], clust_params["distribution"],
                                               clust_params["clust_output_sent_num"], clust_params["isRouge"])
    # output extra noisy documents
    if clust_params["last_noisy"]:
        extra_num = clust_params["clust_output_sent_num"] - not_noisy_num
        noise_num = len(noisy_docs)
        if extra_num > 0:
            noisy_docs = sorted(noisy_docs, key=lambda x: x['score'], reverse=clust_params["isRouge"])
            for i in range(min(noise_num, extra_num)):
                sent_id = noisy_docs[i]["sent_id"]
                sorted_sents_fout.write(sent_lines[sent_id])
                top_id_fout.write(f"{sent_id}\n")
        print(f"Noisy docs total: {len(noisy_docs)} output: {extra_num}")


def sort_and_output_each_clust(ground_scores, clustered_sentences, sent_lines, bl_lines, sorted_sents_fout, top_id_fout,
                               attenuation_coefficient, num_clusters, proportion, distribution, clust_output_sent_num,
                               isRouge):
    """
    :param clustered_sentences: [[dict]]
    :param sent_lines: [str]
    :param sorted_sents_fout: io
    :param num_clusters: int
    :param clust_output_sent_num: int
    :param isRouge: bool
    :return: finish sort and output for each clust
    """
    # sort each cluster
    for i in range(num_clusters):
        clustered_sentences[i] = sorted(clustered_sentences[i], key=lambda x: x['score'], reverse=isRouge)
    # output until clust_output_sent_num
    sent_num = 0
    cluster_out_num = 0
    topic2sent_ids = [[] for k in range(num_clusters)]
    sent_ids = []
    extra_ids = []
    total_bl = 0

    if proportion == 'free':
        while sent_num < clust_output_sent_num:
            for k in range(num_clusters):
                try:
                    sent_id = clustered_sentences[k][cluster_out_num]["sent_id"]
                    topic2sent_ids[k].append(sent_id)
                except IndexError:
                    # this clust is empty or cluster_out_num has exceed its length
                    continue
                sent_num += 1
            cluster_out_num += 1

            if clust_output_sent_num < cluster_out_num:
                print("Too much noise or there is not enough docs: ")
                break
    elif proportion == 'ac' or proportion == 'acr':
        topic_weights = [1] * num_clusters
        topic_sent_nums = [0] * num_clusters
        while total_bl < 1000:
            random_number = random.random()  # [0,1)
            k = 0  # chosen topic
            bar = 0
            weight_sum = sum(topic_weights)
            for topic_id, weight in enumerate(topic_weights):
                bar += weight / weight_sum
                if random_number < bar:
                    k = topic_id
                    break
            # check the topic's sentences
            if topic_sent_nums[k] >= len(clustered_sentences[k]):
                # no enough sentences
                topic_weights[k] = 0
                if sum(topic_weights) == 0:
                    break
            else:
                ground_dict = clustered_sentences[k][topic_sent_nums[k]]
                sent_id = ground_dict["sent_id"]
                topic2sent_ids[k].append(sent_id)
                sent_ids.append(sent_id)
                bl = int(bl_lines[sent_id].strip())
                total_bl += bl
                topic_sent_nums[k] += 1
                # TODO should　attenuation_coefficient be related to length?
                topic_weights[k] *= attenuation_coefficient

    else:
        # proportion is target proportion from training set
        hard_topic = distribution['hard_topic']
        for k in range(num_clusters):
            topic_len_tgt = 1000 * hard_topic[k]
            topic_len = 0
            for ground_dict in clustered_sentences[k]:
                sent_id = ground_dict["sent_id"]
                topic2sent_ids[k].append(sent_id)
                sent_ids.append(sent_id)
                bl = int(bl_lines[sent_id].strip())
                total_bl += bl
                topic_len += bl
                if topic_len > topic_len_tgt:
                    break
    if total_bl < 1000:
        for ground_score in ground_scores:
            if ground_score["sent_id"] not in sent_ids:
                extra_ids.append(ground_score["sent_id"])

    # log and output
    log_str = "Output sent num on each topic: "
    not_noisy_num = 0
    for topic_id, t_sent_ids in enumerate(topic2sent_ids):
        log_str += f"topic{topic_id}: {len(t_sent_ids)} "
        not_noisy_num += len(t_sent_ids)
        for sent_id in t_sent_ids:
            sorted_sents_fout.write(sent_lines[sent_id])
            top_id_fout.write(f"{sent_id}\n")
        # sorted_sents_fout.write(f"<topic{topic_id}>")
        sorted_sents_fout.write(f"<s> ")
    log_str += f"extra: {len(extra_ids)} "
    not_noisy_num += len(extra_ids)
    for extra_id in extra_ids:
        sorted_sents_fout.write(sent_lines[extra_id])
        top_id_fout.write(f"{extra_id}\n")
    print(log_str)
    return not_noisy_num


def out_sorted(ground_scores, sent_lines, bl_lines, tf_lines, pattern_num, sorted_dir, top_bottom_dir, data_id,
               tf_policy, clust_params):
    # all sentences
    sorted_sents_path = os.path.join(sorted_dir, f"{data_id}_sorted.txt")
    sorted_sents_fout = check_fout(sorted_sents_path)
    top_id_path = os.path.join(top_bottom_dir, f"{data_id}_id.txt")
    top_id_fout = check_fout(top_id_path)

    if clust_params['clust_policy'] == 'no':
        out_sents = []
        for ground_score in ground_scores:
            sent_id = ground_score['sent_id']
            if tf_policy == 'not_zero' or tf_policy == 'nz_concat':
                tf = int(tf_lines[sent_id].strip())
                if tf > 0:
                    sorted_sents_fout.write(sent_lines[sent_id])  # do not need extra '\n'
                    out_sents.append(sent_id)
            else:
                # no policy
                sorted_sents_fout.write(sent_lines[sent_id])  # do not need extra '\n'
        if tf_policy == 'nz_concat':
            # concat the rest, in origin order of wikicatsum, which has been sorted by tf_idf
            if len(out_sents) < 20:
                for sent_id, ground_score in enumerate(ground_scores):
                    if sent_id in out_sents:
                        continue
                    sorted_sents_fout.write(sent_lines[sent_id])

        # top and bottom, output full information
        src_num = len(ground_scores)
        max_top_num = pattern_num
        if src_num > max_top_num * 2:
            # just get top and bottom
            top_k_src = ground_scores[:max_top_num]
            bottom_k_src = ground_scores[-max_top_num:]
        else:
            half = src_num // 2
            top_k_src = ground_scores[:half]
            bottom_k_src = ground_scores[half:]
        json_path = os.path.join(top_bottom_dir, f"{data_id}_json.txt")
        write_gold_score_json(top_k_src, bottom_k_src, json_path)
    elif clust_params['clust_policy'] == 'k_means':
        k_means(ground_scores, sent_lines, sorted_sents_fout, top_id_fout, clust_params)
    elif clust_params['clust_policy'] == 'zs_classify':
        # topic of each document is generated by zero shot classify in mask.py
        zs_classify(ground_scores, sent_lines, bl_lines, sorted_sents_fout, top_id_fout, clust_params)


def work():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--tf', type=str, default="no",
                        choices=['no', 'not_zero', 'nz_concat', 'ws_0.5', 'ws_0.25', 'ws_0.75', 'ws_0.88', 'ws_0'],
                        # notice that score = ws * score - (1 - ws) * tf_idf, this suits for ppl score
                        # if is rouge score, then score = ws * score + (1 - ws) * tf_idf
                        help='how to use tf information')
    parser.add_argument('--clust', type=str, default="no",
                        choices=['no', 'k_means', 'zs_classify'],
                        help='ways of clustering')
    parser.add_argument('--clust_num', type=int, default=4,
                        help='how many cluster')
    parser.add_argument('--clust_input_sent_num', type=int, default=128,
                        help='how many sentences that can participate in clustering')
    parser.add_argument('--clust_output_sent_num', type=int, default=64,
                        help='if sent_num < clust_output_sent_num, we output them all and do not perform clustering')
    # prompt for gpt2
    parser.add_argument('--prompt', type=str, default="inverse",
                        choices=['qa', 'inverse', 'none', 'rouge',
                                 'regression_lgb', 'regression_nn',
                                 'regression_lgb_r2_recall', 'regression_nn_r2_recall'],
                        help='ways of prompt')
    parser.add_argument('--rm_each_sent', action='store_true',
                        help='whether score paragraph and remove each sentence')
    parser.add_argument('--para_penal', type=float, default=10.0,
                        help='if use paragraph gain, the penalty of origin para ppl score')
    # data parameters
    parser.add_argument('--lead_section_num', type=int, default=20,
                        help='# of lead section used for generating prompt pattern')
    parser.add_argument('--topic_num', type=int, default=4, help='# of topics')
    parser.add_argument('--last_noisy', action='store_true', help='whether use an extra topic as noise')
    parser.add_argument('--max_logit', action='store_true', help='whether use max or avg logit')
    parser.add_argument('--attenuation_coefficient', type=float, default=0.9,
                        help='if use paragraph gain, the penalty of origin para ppl score')
    parser.add_argument('--proportion', type=str, default="free",
                        # tp means using topic distribution to decide proportion
                        # ac is attenuation_coefficient, r in acr means re_organize by topic order
                        choices=['free', 'tp', 'ac', 'acr'],
                        help='how to decide the proportion of different topics')
    parser.add_argument('--seed', type=int, default=1453, help='random seed')
    parser.add_argument('--addition_pattern_num', type=int, default=4,
                        help='# of additional patterns for inverse prompt')
    # sort parameters
    parser.add_argument('--level', type=str, default="all", choices=['all', 'topic', 'lead_section',
                                                                     'recall', 'precision', 'f1', 'r1_recall',
                                                                     'r2_recall', 'rl_recall'],
                        help='level of sort, first three are choices for qa prompt, last three are for rouge')

    parser.add_argument('--topic_score', type=str, default="min", choices=['avg', 'min'],
                        help='ways of calculating topic score using lead_section scores')
    # path
    parser.add_argument('--split', type=str, default="test", choices=['train', 'valid', 'test'],
                        help='data split')
    parser.add_argument('--start_id', type=int, default=0, help='start id of data, included')
    parser.add_argument('--end_id', type=int, default=2573, help='end id of data, not included')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company', 'wcep'])
    parser.add_argument('--object', default='src', choices=['src', 'tgt'])
    parser.add_argument('--data_dir', type=str, default="/data/tsq/contrastive/clust_documents",
                        help='dir of raw data (after clean noise)')
    parser.add_argument('--tgt_dir', type=str, default="/data/tsq/contrastive/clust_documents/animal/result/",
                        help='dir where processed data will go')
    args = parser.parse_args()
    random.seed(args.seed)
    args.sentences_dir = os.path.join(args.data_dir, args.category, 'raw')
    args.classify_dir = os.path.join(args.data_dir, args.category, 'classify')
    if args.prompt == 'inverse':
        args.pattern_dir = os.path.join(args.data_dir, args.category, 'patterns',
                                        f'inverse_add{args.addition_pattern_num}')
        args.score_dir = os.path.join(args.data_dir, args.category, 'score',
                                      f'inverse_add{args.addition_pattern_num}', f"{args.split}_{args.object}")
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'result',
                                    f'inverse_add{args.addition_pattern_num}', f"{args.split}_{args.object}")
    elif args.prompt == 'qa':
        args.pattern_dir = os.path.join(args.data_dir, args.category, 'patterns',
                                        f'tn{args.topic_num}_lsn{args.lead_section_num}')
        args.score_dir = os.path.join(args.data_dir, args.category, 'score',
                                      f'tn{args.topic_num}_lsn{args.lead_section_num}', f"{args.split}_{args.object}")
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'result',
                                    f'tn{args.topic_num}_lsn{args.lead_section_num}', f"{args.split}_{args.object}")
    else:
        # if prompt is rouge,then this is not a typical prompt, we just need rouge score as a upper bound
        args.pattern_dir = os.path.join(args.data_dir, args.category, f'{args.split}.target.tokenized')
        args.score_dir = os.path.join(args.data_dir, args.category, 'score', args.prompt, f"{args.split}_{args.object}")
        args.tgt_dir = os.path.join(args.data_dir, args.category, 'result', args.prompt, f"{args.split}_{args.object}")

    if args.rm_each_sent:
        args.score_dir = os.path.join(args.score_dir, f'{args.prompt}_score_rm_each_sent')
        args.tgt_dir = os.path.join(args.tgt_dir, f'{args.prompt}_para_penal{args.para_penal}')
    else:
        args.score_dir = os.path.join(args.score_dir, f'{args.prompt}_score')
        args.tgt_dir = os.path.join(args.tgt_dir, f'{args.prompt}_score')

    make_dirs(args.tgt_dir)

    # output sorted sentences under 'result_dir/sorted'
    result_dir = sort_ppl_score(args, args.sentences_dir, args.score_dir, args.pattern_dir, args.tgt_dir, args.split,
                                args.tf, args.clust, args.clust_num, args.clust_input_sent_num,
                                args.clust_output_sent_num, args.proportion, args.attenuation_coefficient,
                                args.start_id, args.end_id, args.level, args.topic_score, args.prompt,
                                args.para_penal, args.rm_each_sent)
    print(f"Finish at {result_dir}")


if __name__ == '__main__':
    work()
