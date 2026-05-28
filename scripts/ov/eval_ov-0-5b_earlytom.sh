#!/bin/bash

MODEL_PATH="/path/to/llava-onevision-qwen2-05b-ov"
TASKS_LIST=("mvbench" "videomme" "egoschema" "longvideobench_val_v")
RETAIN_RATIOS=(0.10 0.15 0.20 0.25)
M=6
EMA=0.9
INNER_k=18
INNER_r=0.5
export HF_ENDPOINT=https://hf-mirror.com


for TASKS in "${TASKS_LIST[@]}"; do
  if [ "$TASKS" == "mvbench" ]; then
    T_LIST=(0.65 0.8 0.8 0.8)
    LAYERS_LIST=("8,14,20" "6,14,20" "6,14,20" "6,14,20")
  elif [ "$TASKS" == "videomme" ]; then
    T_LIST=(0.3 0.4 0.5 0.7)
    LAYERS_LIST=("10,21,23" "8,21,23" "8,21,23" "10,21,23")
  elif [ "$TASKS" == "longvideobench_val_v" ]; then
    T_LIST=(0.3 0.5 0.6 0.8)
    LAYERS_LIST=("10,21,23" "10,21,23" "10,21,23" "4,14,24")
  elif [ "$TASKS" == "egoschema" ]; then
    T_LIST=(0.3 0.5 0.6 0.7)
    LAYERS_LIST=("10,21,23" "10,21,23" "10,21,23" "10,21,23")
  else
    exit 1
  fi

  for i in "${!RETAIN_RATIOS[@]}"; do
    RETAIN_RATIO=${RETAIN_RATIOS[$i]}
    T=${T_LIST[$i]}
    PRUNE_LAYERS=${LAYERS_LIST[$i]}

    export WRAPPER=earlytom
    export RETAIN_RATIO
    export T
    export M
    export INNER_k
    export INNER_r
    export PRUNE_LAYERS

    LOG_DIR="./logs/ov-05b-earlytom/$TASKS/$RETAIN_RATIO"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/ov-05b-earlytom-${RETAIN_RATIO}-t${T}-k${INNER_k}-r${INNER_r}.log"

    echo "==============================="
    echo "Running task=$TASKS"
    echo "retain_ratio=$RETAIN_RATIO, T=$T, M=$M, EMA=$EMA, k=$INNER_k, r=$INNER_r, layers=$PRUNE_LAYERS"
    echo "==============================="

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    accelerate launch --num_processes=8 --main_process_port=25000 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=$MODEL_PATH,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --verbosity=DEBUG \
    --output_path "$LOG_DIR" 2>&1 | tee "$LOG_FILE"
  done
done