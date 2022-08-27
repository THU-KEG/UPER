# path of the fine-tuned checkpoint
CKPT_DIR=/data1/tsq/unilm/result/animal/tf_idf/
CACHE_DIR=/data1/tsq/unilm/pretrain_models/
#kd_weight=0.6
num_training_steps=200000

MODEL_PATH=${CKPT_DIR}ckpt-${num_training_steps}/
#SPLIT=dev
SPLIT=test
# input file that you would like to decode
INPUT_JSON=/data1/tsq/unilm/input/animal/tf_idf/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python src/baseline/NoisySumm/src/decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name ${CACHE_DIR}vocab.txt --input_file ${INPUT_JSON} --split $SPLIT \
  --model_path ${MODEL_PATH} --max_seq_length 1000 --max_tgt_length 120 --batch_size 4 --beam_size 5 \
  --length_penalty 0.7 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "." --min_len 55

#SPLIT=dev
#GOLD_PATH=${DIR}${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
#python evaluations/eval_for_cnndm.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT} --trunc_len 160 --perl
