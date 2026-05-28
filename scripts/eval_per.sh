#!/bin/bash
set -e  # Exit immediately if any command fails

# ==============================================
# Experiment Configuration (Wrapper -> Parameters)
# - aloha (Ours) & holitom: Iterate over RETAIN_RATIO
# - visionzip: Iterate over SPATIAL_TOKENS
# - vanilla: No extra parameters
# ==============================================
declare -A EXPERIMENT_CONFIG=(
  # ["aloha"]=" 0.1 0.15 0.2 0.25"        # Ours: RETAIN_RATIO values
  ["aloha"]="0.2"
  # ["holitom"]="0.1 0.15 0.2 0.25"      # HoliTom: RETAIN_RATIO values
  # ["visionzip"]="20 30 40 50"          # VisionZip: SPATIAL_TOKENS values
  # ["visionzip"]="30" 
  # ["vanilla"]=""                       # Vanilla: No parameters
)

# ==============================================
# Shared Global Config (Applied to all experiments)
# ==============================================
export CUDA_VISIBLE_DEVICES=2  # Fixed GPU ID
export T=0.8                   # Shared threshold for aloha/holitom
# Uncomment below if you need HOLITOM-specific hyperparameters
# export HOLITOM_k=18
# export HOLITOM_r=0.5

export INNER_k=18
export INNER_r=0.5

export HF_ENDPOINT=https://hf-mirror.com

# ==============================================
# Batch Run All Experiments
# ==============================================
for WRAPPER in "${!EXPERIMENT_CONFIG[@]}"; do
  # Set current wrapper
  export WRAPPER
  # Get parameter list for current wrapper
  PARAMS=${EXPERIMENT_CONFIG[$WRAPPER]}

  echo -e "\n========================================"
  echo "Starting Experiment: Wrapper = $WRAPPER"
  echo "========================================"

  # Execute experiments based on wrapper type
  case $WRAPPER in
    aloha|holitom)
      # Run for each RETAIN_RATIO (aloha/holitom)
      for RETAIN_RATIO in $PARAMS; do
        export RETAIN_RATIO
        echo -e "\n--- Current Params: RETAIN_RATIO=$RETAIN_RATIO | T=$T ---"
        python performance.py
      done
      ;;

    visionzip)
      # Run for each SPATIAL_TOKENS (visionzip)
      for SPATIAL_TOKENS in $PARAMS; do
        export SPATIAL_TOKENS
        echo -e "\n--- Current Params: SPATIAL_TOKENS=$SPATIAL_TOKENS ---"
        python performance.py
      done
      ;;
    
    fastvid)
      # Example parameters for fastvid
      for RETAIN_RATIO in $PARAMS; do
        export RETAIN_RATIO
        export DYSET_C=8
        export tau=0.9
        export DTM_P=4
        export DTM_B=0.6
        echo -e "\n--- Current Params: fastvid Mode (DYSET_C=$DYSET_C, tau=$tau, DTM_P=$DTM_P, DTM_B=$DTM_B) ---"
        python performance.py
      done
      ;;

    vanilla)
      # Run vanilla (no extra params)
      echo -e "\n--- Current Params: Vanilla Mode (No extra parameters) ---"
      python performance.py
      ;;
  esac

  echo -e "\n========================================"
  echo "Experiment Completed: Wrapper = $WRAPPER"
  echo "========================================"
done

echo -e "\n🎉 All experiments finished successfully!"