export HF_ENDPOINT=https://hf-mirror.com
MODEL_PATH="/path/to/llava-onevision-qwen2-7b-ov"

# 0.15
WRAPPER=tome merge_num=608 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=$MODEL_PATH,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-tome/0.15 2>&1 | tee ./logs/ov-7b-tome/0.15/ov-7b-tome-0.15.log
