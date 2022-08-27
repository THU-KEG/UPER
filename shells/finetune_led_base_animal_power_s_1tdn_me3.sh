TDN=46773
DOMAIN=animal
ML=16384
ME=2
DP=inverse_add0/ws_0.75/inverse_add_title
RELOAD=/data/tsq/contrastive/clust_documents/animal/bart/test_as_valid/inverse_add0/ws_0.75/inverse_add_title/led-base-16384_fine_tune/few_shot46773_ml16384_mtl256_me3/best_model.pt

export CUDA_VISIBLE_DEVICES=4

python -m src.bart.train --train_data_num ${TDN} \
--min_dec_len 55 \
--max_dec_len 120 \
--beam_size 16 \
--max-len ${ML} \
--model-name allenai/led-base-16384 \
--max_epoch ${ME} \
--reload ${RELOAD} \
--data_path /data/tsq/contrastive/clust_documents/${DOMAIN}/bart/test_as_valid/${DP}


python -m src.baseline.eval_wikicatsum --train_data_num ${TDN} \
--beam_size 16 \
--max_epoch ${ME} \
--max-len ${ML} \
--model-name allenai/led-base-16384 \
--data_path /data/tsq/contrastive/clust_documents/${DOMAIN}/bart/test_as_valid/${DP} \
--max_beam_num 1 \
--duplicate_rate 0.7 \
--reload ${RELOAD} \
--strategy 'split_ref' 'fix_tokenization' 'remove_redundant'
