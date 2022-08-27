​	README



# 1.Module Description 

`prepare.py` -> perform clustering and prepare data for classifier



## classify

This module is used for k-means clustering and the training of Topic Classifier.

### clustering

We use pretained Sentence-Bert to perform sentence-level k-means clustering.

- For wikicatsum:
  - It is reasonable to suppose that the reference summary is composed of several sentences describing an entity from different aspects. So we can gather the similar sentences across different reference summaries to generate clusters. We call the cluster 'topic'.  
- For Xsum:
  - `update 20 June` : The target reference summary of Xsum dataset usually contains only 1 sentence. So it is not comprehensive summary. It is not reasonable to divide a single sentence into different topics.
  - Maybe we can have a try. Divide src document into different topics and predict the only one topic for generating summary.

## label

This module can label each sentence in source document and target summary.

After labeling, we will also tokenize the text with `spacy` to output the `%split.source.tokenized` and `%split.target.tokenized` which will be used in `SimCLS`

We implement three ways of labeling the sentence with topic label:

- `make_labels.py` is used for labeling source document and target summary by Topic Classifier which is an albert-based model trained in classify module.
  - So you should train the Topic Classifier before using it.
- `make_labels_zero_shot.py` is used for labeling source document and target summary by pretained Sentence-Bert model. We perform labeling with a zero-shot setting which means we will utilize the title of each topic to count similarity between topic title and sentence. So the sentence is labeled with the closest meaning topic title.
  - You don't need to train any classifier because we perform in zero shot way.
- `make_labels_random.py` is used for labeling source document and target summary randomly. It will only use some regex to filter the noisy sentences. Other sentences will be labeled randomly.
  - You don't need to train any classifier because we perform in random.



## bart

This module can generate candidate summary with Bart model.



## SimCLS

This module can evaluate the candidate summaries by a Roberta model which is trained with contrastive learning.



## combine

This module can combine the candidate to form the final summary.





# 2. How to Run



## For wikicatsum

1. k-means  clustering and prepare for classifier training

>  If you use --num_clusters 3, you will find the final generated corpus has 4 topic. This is right because the extra topic is an additional NOISE topic whose sentences are found by regex.

```
CUDA_VISIBLE_DEVICES=5 nohup python -m src.prepare --dataset wikicatsum \
--learning cluster \
--num_clusters 3 \
--data_dir /data1/tsq/contrastive/clustering_3/raw_dataset/ \
--classifier_dir /data1/tsq/contrastive/clustering_3/classifier/ \
--generator_dir /data1/tsq/contrastive/clustering_3/generator/ \
--tokenizer_dir /data1/tsq/TWAG/data/pretrained_models/albert_tokenizer  > cluster_prepare_619.log  &
```

2. train albert topic classifier ( use animal category for example)

```
CUDA_VISIBLE_DEVICES=4 nohup python -m src.classify.train  --category animal \
--classifier-dir /data1/tsq/contrastive/clustering_3/classifier/ \
--albert-model-dir /data1/tsq/TWAG/data/pretrained_models/albert_model \
--max-len 100 \
--topic-num 4 \
--lr 3e-5 > classify_train_animal_619.log &
```

3. label the src document and tgt document ( use animal category for example)

> If use flag `--old_label` , we will use the existing `label.pkl`  and skip the process of labeling.
>
> I put it here because I have generated  `label.pkl`   before.

```
CUDA_VISIBLE_DEVICES=5 nohup python -m src.label.run --classifier_dir /data1/tsq/contrastive/clustering_3/classifier/ \
--generator_dir /data1/tsq/contrastive/clustering_3/generator/ \
--tokenizer_dir /data1/tsq/TWAG/data/pretrained_models/albert_tokenizer \
--albert-model-dir /data1/tsq/TWAG/data/pretrained_models/albert_model \
--category animal  > run_label_animal_620.log &
```

4. generate candidate summary for every topic of a sample document

> The `run.py` has not been written yet, the choice of `sentence_level` or `summary_level` candidate generation will be implemented in the future. Now we only provide the summary_level candidates by running commands in `./bart/bart_README.md`.
>
> See `./bart/bart_README.md` for more information.
>
> First, we should process the dataset:

```
nohup python -m src.bart.preprocess --topicdata-dir /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/ \
--num_topics 3 > preprocess_bart_animal_621_debug.log &
```

> Second, fine-tune bart_base model on each topic, take topic0 for example:

```
CUDA_VISIBLE_DEVICES=6 nohup python -m src.bart.train --data_path /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/topic0/ \
--beam_size 4 > train_bart_animal_t0_623.log &
```

> Third, generate candidates using beam search strategy, take topic0 for example:

```
CUDA_VISIBLE_DEVICES=5 nohup python -m src.bart.gen_candidate --topicdata-dir /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/ \
--num_topics 3 \
--ckpt /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/topic0/bart-base_fine_tune/model_epoch2_val0.379.pt \
--single_topic_id 0 > gen_candidate_bart_base_ft_animal_t0_625.log &
```





5. use the simcls for our task

>  preprocess:  We will process topic data under `topicdata-dir`  separately, take topic0 for example

```
python -m src.SimCLS.preprocess --topic_dir /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/topic0/ \
--tgt_dir_name diverse \
--split all \
--cand_num 16
```

>  train: Every topic can train separately, take topic0 for example

```
nohup python -m src.SimCLS.main --cuda \
--gpuid 5 \
-l \
--dataset /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/topic0 > train_simcls_animal_t0_627.log &
```

> Evaluation: after training, we can output the best summary candidate reranked by simcls.
>
> The model_pt is the checkpoint saved during training. 

```
nohup python -m src.SimCLS.main --model_pt 21-06-27-13 \
--cuda \
--gpuid 6 \
-l \
-e \
--dataset /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_clust3/topic0 > evaluate_simcls_animal_t0_627.log &
```

6. combine the result

> There are many ways of combine, including:  origin、max、random、min、rerank
>
> `combine/oracle.py` can handle origin、max、random、min. You just need to specify the `--combine_mode`  flag

```
python -m src.combine.oracle --split test \
--category animal \
--generator_dir /data1/tsq/contrastive/clustering_3/generator/ \
--clust_num 3 \
--combine_mode origin
```

> `combine/test.py` can handle rerank

```
nohup python -m src.combine.test --category animal \
--split test \
--generator_dir /data1/tsq/contrastive/clustering_3/generator/ \
--clust_num 3 > combine_simcls_animal_627.log &
```

