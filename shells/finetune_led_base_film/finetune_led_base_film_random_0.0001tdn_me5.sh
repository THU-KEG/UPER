TDN=5
DOMAIN=film
ML=16384
ME=5
DP=random

export CUDA_VISIBLE_DEVICES=5

python -m src.bart.train --train_data_num ${TDN} \
--min_dec_len 55 \
--max_dec_len 120 \
--beam_size 16 \
--max-len ${ML} \
--model-name allenai/led-base-16384 \
--max_epoch ${ME} \
--data_path /data/tsq/contrastive/clust_documents/${DOMAIN}/bart/test_as_valid/${DP}


python -m src.baseline.eval_wikicatsum --train_data_num ${TDN} \
--beam_size 16 \
--max_epoch ${ME} \
--max-len ${ML} \
--model-name allenai/led-base-16384 \
--data_path /data/tsq/contrastive/clust_documents/${DOMAIN}/bart/test_as_valid/${DP} \
--max_beam_num 1 \
--duplicate_rate 0.7 \
--strategy 'split_ref' 'fix_tokenization' 'remove_redundant'
