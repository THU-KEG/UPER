# path of training data
TRAIN_FILE=/data1/tsq/unilm/input/animal/train.json
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=/data1/tsq/unilm/result/animal
# folder used to cache package dependencies
CACHE_DIR=/data1/tsq/unilm/pretrain_models

export CUDA_VISIBLE_DEVICES=0,6
python -m torch.distributed.launch --nproc_per_node=2 run_seq2seq.py \
                                   --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
                                   --model_type minilm --model_name_or_path minilm-l12-h384-uncased \
                                   --do_lower_case --fp16 --fp16_opt_level O2 --max_source_seq_length 1000 --max_target_seq_length 120 \
                                   --per_gpu_train_batch_size 2 --gradient_accumulation_steps 1 \
                                   --learning_rate 1e-4 --num_warmup_steps 500 --num_training_steps 108000 --cache_dir ${CACHE_DIR}
