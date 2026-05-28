export WRAPPER=aloha 
export RETAIN_RATIO=0.15
export T=0.8
export M=12
export INNER_k=18 
export INNER_r=0.5 
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=6
echo $WRAPPER

python example.py