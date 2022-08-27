DOMAIN="animal"
TEXT="/data/tsq/wikicatsum/"$DOMAIN
TRAIN_NUM=46773
TEST_NUM=2573
GPUID=4
DATE="105"
I=1024
O=512
MTN=4096
PROP=free

python -m src.clust.gather --split train --category $DOMAIN --start_id 0 --end_id $TRAIN_NUM --addition_pattern_num 0 --prompt inverse --tf ws_0.75 --clust no --topic_num 4 --lead_section_num 20 --clust_input_sent_num $I --clust_output_sent_num $O  --proportion $PROP --attenuation_coefficient 0.9

python -m src.clust.gather --split test --category $DOMAIN --start_id 0 --end_id $TEST_NUM --addition_pattern_num 0 --prompt inverse --tf ws_0.75 --clust no --topic_num 4 --lead_section_num 20 --clust_input_sent_num $I --clust_output_sent_num $O  --proportion $PROP --attenuation_coefficient 0.9 > gather_$DOMAIN_test_ln20_ws0.75_k4_i$I_$PROP0.9_max_$DATE.log

CUDA_VISIBLE_DEVICES=$GPUID python -m src.clust.test --split train --category $DOMAIN --start_id 0 --end_id $TRAIN_NUM --addition_pattern_num 0 --prompt inverse --strategy 'no' --max_read_lines $O --max_token_num $MTN --tf ws_0.75 --clust no --clust_num 4 --topic_num 4 --lead_section_num 20 --clust_input_sent_num $I --clust_output_sent_num $O  --proportion $PROP --attenuation_coefficient 0.9 --keep_special_token > ttrain_animal_pi_ln20_ws0.75_$O_$PROP_0.9_max_$DATE.log

CUDA_VISIBLE_DEVICES=$GPUID python -m src.clust.test --split test --category $DOMAIN --start_id 0 --end_id $TEST_NUM --addition_pattern_num 0 --prompt inverse --strategy 'no' --max_read_lines $O --max_token_num $MTN --tf ws_0.75 --clust no --clust_num 4 --topic_num 4 --lead_section_num 20 --clust_input_sent_num $I --clust_output_sent_num $O  --proportion $PROP --attenuation_coefficient 0.9 --keep_special_token > ttest_animal_ln20_ws0.75_k4_i_$I_$PROP_0.9_max_$DATE.log

CUDA_VISIBLE_DEVICES=$GPUID python -m src.clust.extract --addition_pattern_num 0 --prompt inverse --strategy 'no' --max_read_lines $O --max_token_num $MTN --split_mode test_as_valid --category $DOMAIN --tokenizer-dir facebook/bart-base --max-len 1000 --add_title --tf ws_0.75 --clust no --clust_num 4 --topic_num 4 --lead_section_num 20 --clust_input_sent_num $I --clust_output_sent_num $O  --proportion $PROP --attenuation_coefficient 0.9  > preprocess_bartb_animal_ln20_ws0.75_k4_$I_$PROP_0.9_max_$DATE.log
