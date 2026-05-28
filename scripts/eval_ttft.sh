export WRAPPER=holitom 
export RETAIN_RATIO=0.15 
export T=0.80
export M=6
export HOLITOM_k=18
export HOLITOM_r=0.5

export CUDA_VISIBLE_DEVICES=2

nsys profile \
    -t nvtx \
    -o baseline.nsys-rep \
    --force-overwrite true \
    python example.py