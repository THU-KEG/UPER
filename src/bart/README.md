Bart_README



This module can generate candidate summary with Bart model.



# Data path

root dir of topic data:

```
os.path.join(args.generator_dir, category, "topic_data_{}clust".format(num_topics))
```

e,g:

/data1/tsq/contrastive/clustering_3/generator/animal/topic_data_5clust

under this, there will be `num_topics` subdir:

- topic0/
- topic1/
- ...
- topic{num_topics}/



under topic{i}, there will be 12 files:(use test split for example)

- test.source.tokenized
- test.target.tokenized
- test.source
- test.target



# Preprocess

use `preprocess.py`

```
nohup python -m src.bart.preprocess --topicdata-dir /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_3clust/ \
--num_topics 3 > preprocess_bart_animal.log &
```

> In fact, the `make_labels.py`  has performed tokenize by AlbertTokenizer.
>
> I don't know whether bart can use the result from AlbertTokenizer directly.
>
> So I just tokenize the sentences again using BartTokenizer

# Fine-Tune

We can fine-tune by counting loss between generated summary and reference summary.

[example from github](https://github.com/ohmeow/ohmeow_website/blob/master/_notebooks/2020-05-23-text-generation-with-blurr.ipynb)



# Candidate summary generation

use `gen_candidate.py`

- use fine-tuned model

  - use `--ckpt` 

- use raw Bart model

  - use `--raw_model`

  - ```
    nohup python -m src.bart.gen_candidate --topicdata-dir /data1/tsq/contrastive/clustering_3/generator/animal/topic_data_3clust/ \
    --num_topics 3 \
    --raw_model > gen_candidate_bart_animal.log &
    ```

    

