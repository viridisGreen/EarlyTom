
MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model qwen2_5_vl \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--model_args pretrained=$MODEL_PATH,min_pixels=37632,max_pixels=200704,max_num_frames=32,interleave_visuals=False \
--batch_size 1 \
--log_samples \
--output_path ./logs/qwen2_5_vl/0.10 2>&1 | tee ./logs/qwen2_5_vl/0.10/qwen2_5_vl-32f.log
