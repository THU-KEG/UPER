README_CLUST



# Preprocess

## sentencize

input: raw data, ignore index file

output: 

- sentencized document txt , file name: `{data_num}.src ` , each line is a sentence
- title of each data

note: each sentence can be found by a unique id.

- sentence with prefix space are tokenized, too.
  - saved as `sentence_prefix_space.pt`
  - note: should strip() when reading lines

> cmd:
>
> ```
> python -m src.clust.preprocess --mode sentencize --category animal --max_sent_len 64 --para_sent_num 3
> ```

## generate_pattern

input:  topic, lead_section_num, titles

output: pkl of processed tensor

- patterns of each topic are first tokenized by `GPT2TokenizerFast`
  - saved as `tn{topic_num}_lsn{lead_section_num}/{split}_pattern.pt` 

> note: must finish sentencize before using generate_pattern
>
> ```
> python -m src.clust.preprocess --mode generate_pattern --category animal --topic_num 4 --lead_section_num 10 
> ```
>
> 

## align_sent_para

input is same as sentencize

output:

- sent2para_list,  list of dict
  -  dict: key is sent_id, value is para_id
  - index of list is data_id

- para2sents_list
  - index of list is data_id
  -  dict: key is para_id, value is [sent_id]

> note: must finish sentencize before using align_sent_para
>
> ```
> python -m src.clust.preprocess --mode align_sent_para --category animal --max_sent_len 64 --para_sent_num 3
> ```

# Score

use gpt2 perplexity to score sentences

## calculate

input:  patterns, sentences

output: perplexity_list of each sentence

- concat prompt pattern and sentence to calculate perplexity
- truncate the total tensor length (shape[1]) to 1024 before feeding it into `gpt-2`
- use a start_id and end_id to implement multi_processing

### 3 kinds of ppl scores

- qa
  - use question pattern to prompt the gpt2 
  - score every sentence with different questions
- none
  - use no prompt pattern
  - score every sentence directly by perplexity
- inverse
  - score every document(paragraph) with ppl score and a fixed prompt pattern suffix to the document like: `This document is about ${title}.`
  - for every sentences in a document, **remove** it then score the document, so we can get ppl score gain



> cmd:
>
> ```
> python -m src.clust.score --split test --category animal --start_id 0 --end_id 625 --topic_num 4 --lead_section_num 10 --prompt qa
> ```

# Gather

gather different sentences using ppl scores

## sort_ppl_score

input:  ppl scores, sentences

output: a kind of sort of all sentences

We can sort them on different levels: 

- sort by avg of all scores
- sort by each lead section
- sort by each topic
  - topic score can be the avg of lead section scores
  - topic score can be min of lead section scores

When sorting, we can use a dict like {scores: [[int]], sent_id: int} and perform different sort ways on scores.



> cmd:
>
> ```
> python -m src.clust.gather --split test --category animal --start_id 0 --end_id 625 --topic_num 4 --lead_section_num 10 --level topic --topic_score min
> ```



# Test

generate final summary and  count rouge

## gen_final_summary

input: sorted sentences, strategy, max_read_lines, max_token_num, 

output: final_summary

strategy is used for truncate and filter sentences

> cmd:
>
> ```
> python -m src.clust.test --split test --category animal --start_id 0 --end_id 625 --topic_num 4 --lead_section_num 10 --level topic --topic_score min --prompt qa --strategy ['filter_syntactic_err', 'stop_sentences', 'remove_redundant'] --max_read_lines 20 --max_token_num 200
> ```