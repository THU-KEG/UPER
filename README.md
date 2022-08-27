# Introduction
This repository stores the code for the COLING22 paper "UPER: Boosting Multi-Document Summarization with an Unsupervised Prompt-based Extractor"

We use the perplexity calculated by GPT2 to evaluate the semantic salience of source documents in MDS datasets. 

This metric can be applied on the extractive stage of the extract-then-abstract paradigm.



# Code routing

The extractive stage is implemented by `src/clust`.

The abstractive stage(BART and LED) is implemented by `src/bart`.

We use the code in `src/statistics` to draw graphics in our paper.

The baselines we compare are in `src/baseline`.

Our training scripts for the abstractive stage are available in `shells/`.

The human evaluation results are saved in the excel tables under `human_evaluation/`.

# Get Start

First, clone our code from github:

```
git clone https://github.com/THU-KEG/UPER.git
```

Then, Enter UPER's root directory. All command then should be executed here.

```
cd UPER
```

The python libraries can be installed by:

```
pip install requirements.txt
```

Finally, prepare the data:

- The WCEP dataset can be download at this [website](https://github.com/complementizer/wcep-mds-dataset).
- The WikiCatSum dataset can be download at [google drive](https://drive.google.com/file/d/1gw_j_3rF38boFaTurCrHR4MMqJAc7-CU/view?usp=sharing). The size of this dataset is 6.9 GB (unzipped).



# Extractive-Stage Preprocess

For example, if we want to get the process wcep dadaset, then the commands are:

1. sentencize:

```
python -m src.clust.preprocess_wcep --mode sentencize --category wcep --max_sent_len 64 --para_sent_num 3
```

2. generate_pattern:

```
python -m src.clust.preprocess_wcep --mode generate_pattern --category wcep --prompt inverse
```

3. scoring perplexity:

```
CUDA_VISIBLE_DEVICES=0 nohup python -m src.clust.score --split train --category wcep --start_id 0 --end_id 8158 --addition_pattern_num 4 --prompt inverse > score_wcep_train_pn_pi_apn4_418_1.log &

CUDA_VISIBLE_DEVICES=7 nohup python -m src.clust.score --split test --category wcep --start_id 0 --end_id 1022 --addition_pattern_num 4 --prompt inverse > score_wcep_test_pn_pi_apn4_418_1.log &

CUDA_VISIBLE_DEVICES=5 nohup python -m src.clust.score --split val --category wcep --start_id 0 --end_id 1020 --addition_pattern_num 4 --prompt inverse > score_wcep_val_pn_pi_apn4_418_3.log &
```

4. tf_idf:

```
python -m src.clust.preprocess_wcep --mode output_tf_idf --category wcep
```

5. normalize:

```
python -m src.clust.preprocess_wcep --mode normalize_tf_idf --category wcep
```

6. combine with perplexity and extract the final result：

```
python -m src.clust.gather --split train --category wcep --start_id 0 --end_id 8158 --addition_pattern_num 4 --prompt inverse --tf ws_0.75 --clust no

python -m src.clust.gather --split test --category wcep --start_id 0 --end_id 1022 --addition_pattern_num 4 --prompt inverse --tf ws_0.75 --clust no 

CUDA_VISIBLE_DEVICES=7 nohup python -m src.clust.test --split train --category wcep --start_id 0 --end_id 8158 --addition_pattern_num 4 --prompt inverse --strategy 'no' --max_read_lines 512 --max_token_num 16384 --tf ws_0.75 --clust no > ttrain_wcep_apn0_ws0.75_cno_419.log &

CUDA_VISIBLE_DEVICES=7 nohup python -m src.clust.test --split test --category wcep --start_id 0 --end_id 1022 --addition_pattern_num 4 --prompt inverse --strategy 'no' --max_read_lines 512 --max_token_num 16384 --tf ws_0.75 --clust no > ttest_wcep_apn0_ws0.75_cno_419.log &

CUDA_VISIBLE_DEVICES=0 nohup python -m src.clust.extract --addition_pattern_num 4 --prompt inverse --strategy 'no' --max_read_lines 512 --max_token_num 16384 --split_mode test_as_valid --category wcep --tokenizer-dir facebook/bart-base --max-len 16384 --tf ws_0.75 --add_title --clust no > preprocess_led_wcep_apn0_ws0.75_at_cno_421.log &
```

# Abstractive-Stage

You can directly use the scripts in `shells/`, which contains the training and testing of BART and LED models.

For more information of BART, you can refer to the [huggingface doc](https://huggingface.co/docs/transformers/model_doc/bart).

For more information of LED, you can also refer to this [huggingface's doc](https://huggingface.co/docs/transformers/model_doc/led).

Note that the LED is a very large model, which costs us one RTX 3090 with about 20GiB memory to run.



