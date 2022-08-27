CAT=company
EXT=tf_idf

CUDA_VISIBLE_DEVICES=0 nohup python -m src.baseline.eval_wikicatsum --category ${CAT} \
--model-name unilm \
--data_path /data1/tsq/unilm/result/${CAT}/${EXT}/ckpt-400000  \
--max_beam_num 1 \
--duplicate_rate 0.7 \
--strategy 'split_ref' 'fix_tokenization' 'remove_redundant' > eval_${CAT}_${EXT}_s3_dr0.7_unilm_ckpt400000_106.log &

CUDA_VISIBLE_DEVICES=0 nohup python -m src.baseline.eval_wikicatsum --category ${CAT} \
--model-name unilm \
--data_path /data1/tsq/unilm/result/${CAT}/${EXT}/ckpt-200000  \
--max_beam_num 1 \
--duplicate_rate 0.7 \
--strategy 'split_ref' 'fix_tokenization' 'remove_redundant' > eval_${CAT}_${EXT}_s3_dr0.7_unilm_ckpt200000_827.log &
