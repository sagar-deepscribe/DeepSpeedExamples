#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_test
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed   --include localhost:0,1,2,3 --master_addr=127.0.0.1 --master_port=29532 /home/sagar/DeepScribeML/scripts/rlhf/training/step1_supervised_finetuning/main.py \
   --data_path /ml_dev/users/dong/data/unify/0.2.1 \
   --data_split 2,4,4 \
   --model_name_or_path /ml_dev/users/sagar/models/falcon-40b \
   --tokenizer_name_or_path /ml_dev/users/sagar/models/falcon-40b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed deepspeed_config.json\
   --output_dir $OUTPUT
   # \
   &> $OUTPUT/training.log
