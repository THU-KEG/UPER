README_RECALL

> This package is used for classifying whether a sentence has highly ROUGE recall score.

# Label

> We use `label.py` to count ROUGE score between every source document sentence and reference summary. Then the top-k sentence will be labeled 1, others 0. The top-k sentences will be truncated to 500 tokens and feed into Abstractive model to get a ext-abs result. We call this extractive way as `cheating`.

input: src .  ref

output: score and label of every sentence , extractive result in  `cheating` order (for both train set and test set) as a csv. We will use the csv data for the training of regression or binary classification task.

so we will output a json, with origin_sent_id and sorted in cheating or just label them and use old code to sort?

In fact, use rouge score is an upper bound, Like this:

```
CUDA_VISIBLE_DEVICES=5 nohup python -m src.clust.score --split test --category animal --start_id 0 --end_id 2573 --prompt rouge --level f1
```

Then, we can label the data with rouge labels, like this:

```
CUDA_VISIBLE_DEVICES=5 nohup python -m src.clust.recall.label --task regression --label f1 --train_num 500 --category animal 
```

# lgb

We can use lightGBM to train a regression model and predict the score for every sentence