# path of training data
TRAIN_FILE=/data/tsq/unilm/input/taxonomy/train.json
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=/data/tsq/unilm/result/taxonomy/
# folder used to cache package dependencies
CACHE_DIR=/data/tsq/unilm/pretrain_models/

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=4 src/baseline/NoisySumm/src/run_seq2seq.py \
                                   --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
                                   --model_type unilm --model_name_or_path ${CACHE_DIR}pytorch_model.bin \
                                   --config_name ${CACHE_DIR}config.json --tokenizer_name ${CACHE_DIR}vocab.txt \
                                   --fp16 --fp16_opt_level O2 --max_source_seq_length 1000 --max_target_seq_length 8 \
                                   --per_gpu_train_batch_size 2 --gradient_accumulation_steps 1 \
                                   --learning_rate 3e-5 --num_warmup_steps 50 --num_training_steps 30000  --cache_dir ${CACHE_DIR} --save_steps 3000
