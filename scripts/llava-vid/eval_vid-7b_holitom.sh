MODEL_PATH="/path/to/LLaVA-Video-7B-Qwen2"
export HF_ENDPOINT=https://hf-mirror.com


WRAPPER=holitom RETAIN_RATIO=0.15 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=$MODEL_PATH,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/0.15 2>&1 | tee ./logs/vid-7b-holitom/0.15/vid-7b-holitom-0.15-t0.80-k18-r0.5.log

WRAPPER=holitom RETAIN_RATIO=0.10 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=$MODEL_PATH,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/0.10 2>&1 | tee ./logs/vid-7b-holitom/0.10/vid-7b-holitom-0.10-t0.80-k18-r0.5.log

WRAPPER=holitom RETAIN_RATIO=0.20 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=$MODEL_PATH,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/0.20 2>&1 | tee ./logs/vid-7b-holitom/0.20/vid-7b-holitom-0.20-t0.80-k18-r0.5.log

WRAPPER=holitom RETAIN_RATIO=0.25 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=$MODEL_PATH,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/0.25 2>&1 | tee ./logs/vid-7b-holitom/0.25/vid-7b-holitom-0.25-t0.80-k18-r0.5.log