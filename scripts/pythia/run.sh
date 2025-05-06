#!/bin/bash

# help string
HELP="Usage: run.sh -m model -d dataset -s size -l layers [-t on|off]"

#
# command-line arguments
#
TOKENIZED=""  # default embedding pooling is none
while getopts ":f:m:d:s:l:t:" option; do
   case $option in
      m) # set model
        MODEL=$OPTARG;;
      d) # set data
        TASK=$OPTARG;;
      s) # set LM size
        LM_SIZE=$OPTARG;;
      l) # set LM size
        LM_LAYERS=$OPTARG;;
      t) # set data tokenization
        TOKENIZED="--tokenized ";;
     \?) # Invalid option
         echo "[Error]: Unknown argument."
         echo "${HELP}"
         exit 1;;
   esac
done

# sanity check whether all necessary arguments were provided
if [ -z "${MODEL}" ]; then
  echo "[Error] Please set a model."
  echo "${HELP}"
  exit 1
fi
if [ -z "${TASK}" ]; then
  echo "[Error] Please set a dataset."
  echo "${HELP}"
  exit 1
fi
if [ -z "${LM_SIZE}" ]; then
  echo "[Error] Please specify a model size."
  echo "${HELP}"
  exit 1
fi
if [ -z "${LM_LAYERS}" ]; then
  echo "[Error] Please specify the number of model layers."
  echo "${HELP}"
  exit 1
fi

#
# global variables
#
# data variables
DATA_ROOT=data/pythia
RAW_DATA_PATH=$DATA_ROOT/original/$TASK
ALIGNED_DATA_PATH=$DATA_ROOT/aligned/$TASK
VALID_SPLIT="dev"
# model setup
LM_ROOT="EleutherAI/pythia-${LM_SIZE}"


# experiment setup
EXP_PATH=exp/pythia/${LM_SIZE}/"${MODEL//[()\/]}"/$TASK
SEEDS=( 0 1 2 3 4 5 6 7 8 9 )
STEPS=( 0 1 2 4 8 16 32 64 128 256 512 1000 )
STEPS+=( $(seq 10000 10000 140000) )
STEPS+=( 143000 )
LAYERS="auto(${LM_LAYERS})"
BATCH_SIZE=64
LEARNING_RATE=0.005
DECAY_RATE=0.0
MAX_EPOCHS=30
# statistics
NUM_EXP=0
NUM_RUN=0
NUM_ERR=0

# model-specific configurations
if [[ "${LM_SIZE}" == "410m" ]]; then
  BATCH_SIZE=32
fi

# task-specific configurations
if [[ "${TASK}" == "topic" ]]; then
  BATCH_SIZE=$((BATCH_SIZE / 4))
fi

#
# Helper Functions
#

preprocess() {
  # function arguments
  split=$1
  lm_base=$2
  lm_seed=$3

  output_directory="${ALIGNED_DATA_PATH}/seed${lm_seed}"
  output_file="${output_directory}/${split}.csv"
  # check if pre-processed file already exists
  if [ -f "${output_file}" ]; then
    echo "[Warning] Pre-processed data at '${output_file}' already exists. Not re-processing."
  else
    if [ -f "${RAW_DATA_PATH}/${split}.csv" ]; then
      # create aligned data directory if not present
      if [ ! -d "${output_directory}" ]; then
        mkdir -p "${output_directory}"
      fi

      # apply label repetition according to canonical tokenizer
      if [ "${lm_seed}" -eq 0 ]; then
        python preprocess.py \
          --data-path "${RAW_DATA_PATH}/${split}.csv" \
          --output-path "${output_file}" \
          --model-name "${lm_base}" \
          --repeat-labels \
          $TOKENIZED
      else
        python preprocess.py \
        --data-path "${RAW_DATA_PATH}/${split}.csv" \
        --output-path "${output_file}" \
        --model-name "${lm_base}-seed${seed}" \
        --model-revision "step0" \
        --repeat-labels \
        $TOKENIZED
      fi

    else
      echo "[Warning] No ${split}-split exists at ${RAW_DATA_PATH}. Skipped."
    fi
  fi
}

train() {
  # function arguments
  step=$1
  seed=$2

  # set up language model identifier
  if [ "$seed" -gt 0 ]; then
    lm_base="${LM_ROOT}-seed${seed}"
  fi
  lm_id="${lm_base#*/}"

  # set up data paths
  train_path="${ALIGNED_DATA_PATH}/seed${seed}/train.csv"
  valid_path="${ALIGNED_DATA_PATH}/seed${seed}/dev.csv"

  # check if experiment exists
  exp_dir="${EXP_PATH}/seed${seed}-step${step}"
  if [ -f "$exp_dir/best.pt" ]; then
    echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
  # start training
  else
    # delete any existing experiment files (w/o best.pt)
    if [ -d "${exp_dir}" ] && [ ! -f "$exp_dir/best.pt" ]; then
      rm -r "${exp_dir}"
      echo "Deleted existing incomplete experiment at '${exp_dir}'."
    fi
    echo "Training '${lm_id}' with random seed ${seed}."

    # run training
    python classify.py \
      --train-path "${train_path}" \
      --valid-path "${valid_path}" \
      $TOKENIZED \
      --model-name "${lm_base}" \
      --model-revision "step${step}" \
      --layers "${LAYERS}" \
      --classifier "${MODEL}" \
      --exp-path "${exp_dir}" \
      --epochs $MAX_EPOCHS \
      --batch-size $BATCH_SIZE \
      --learning-rate $LEARNING_RATE \
      --decay-rate $DECAY_RATE \
      --random-seed $seed

    # check for error
    if [ $? -ne 0 ]; then
      echo "[Error] Could not complete training of previous model."
      (( NUM_ERR++ ))
    fi
    (( NUM_RUN++ ))
  fi
}

evaluate() {
  # function arguments
  step=$1
  seed=$2

  # set up language model identifier
  if [ "$seed" -gt 0 ]; then
    lm_base="${LM_ROOT}-seed${seed}"
  fi
  lm_id="${lm_base#*/}"

  # set up data paths
  train_path="${ALIGNED_DATA_PATH}/seed${seed}/train.csv"
  valid_path="${ALIGNED_DATA_PATH}/seed${seed}/${VALID_SPLIT}.csv"
  ood_valid_path="${ALIGNED_DATA_PATH}/seed${seed}/${VALID_SPLIT}-ood.csv"

  # check if experiment exists
  exp_dir="${EXP_PATH}/seed${seed}-step${step}"
  if [ -f "$exp_dir/train-stats.json" ]; then
    echo "[Warning] Training predictions in '$exp_dir' already exist. Not re-predicting."
  # compute final training statistics
  else
    echo "Evaluating '${lm_id}' on ${train_path}."
    python classify.py \
      --valid-path "${train_path}" \
      $TOKENIZED \
      --model-name "$lm_base" \
      --model-revision "step${step}" \
      --layers "${LAYERS}" \
      --classifier "${MODEL}" \
      --exp-path "${exp_dir}" \
      --batch-size $BATCH_SIZE \
      --random-seed $seed \
      --prediction
    # delete training predictions to preserve space (stats are sufficient for computing codelength)
    rm "$exp_dir/train-pred-0.csv"
  fi

  # run prediction on validation data
  if [ -f "$exp_dir/${VALID_SPLIT}-pred-0.csv" ]; then
    echo "[Warning] Validation predictions in '$exp_dir' already exist. Not re-predicting."
  else
    echo "Evaluating '${lm_id}' on ${valid_path}."
    python classify.py \
      --valid-path "${valid_path}" \
      $TOKENIZED \
      --model-name "$lm_base" \
      --model-revision "step${step}" \
      --layers "${LAYERS}" \
      --classifier "${MODEL}" \
      --exp-path "${exp_dir}" \
      --batch-size $BATCH_SIZE \
      --random-seed $seed \
      --prediction
    # check for error
    if [ $? -ne 0 ]; then
      echo "[Error] Could not complete evaluation of previous model."
      (( NUM_ERR++ ))
    fi
    (( NUM_RUN++ ))
  fi

  # compute and store metrics
  stats_path="${exp_dir}/${VALID_SPLIT}-results.json"
  if [ -f "$stats_path" ]; then
    echo "[Warning] Evaluation results in '$stats_path' already exist. Not re-evaluating."
  else
    # compute evaluation metrics
    python scripts/eval/classification.py \
      --target "${valid_path}" \
      --prediction "$exp_dir/$VALID_SPLIT-pred-0.csv" \
      --output "$stats_path"
  fi

  # check if OOD evaluation data is available
  if [ -f "${ood_valid_path}" ]; then
    # run OOD prediction
    if [ -f "$exp_dir/${VALID_SPLIT}-ood-pred-0.csv" ]; then
      echo "[Warning] OOD validation predictions in '$exp_dir' already exist. Not re-predicting."
    else
      echo "Evaluating '${lm_id}' on ${ood_valid_path}."
      python classify.py \
        --valid-path "${ood_valid_path}" \
        $TOKENIZED \
        --model-name "$lm_base" \
        --layers "${LAYERS}" \
        --classifier "${MODEL}" \
        --exp-path "${exp_dir}" \
        --batch-size $BATCH_SIZE \
        --random-seed $seed \
        --prediction
      # check for error
      if [ $? -ne 0 ]; then
        echo "[Error] Could not complete evaluation of previous model."
        (( NUM_ERR++ ))
      fi
      (( NUM_RUN++ ))
    fi
    # compute and store metrics
    if [ -f "${exp_dir}/${VALID_SPLIT}-ood-results.json" ]; then
      echo "[Warning] Evaluation results in '$stats_path' already exist. Not re-evaluating."
    else
      # compute evaluation metrics
      python scripts/eval/classification.py \
        --target "${ood_valid_path}" \
        --prediction "$exp_dir/$VALID_SPLIT-ood-pred-0.csv" \
        --output "${exp_dir}/${VALID_SPLIT}-ood-results.json"
    fi
  fi
}

#
# Main
#

echo "=============================================================="
echo "❄️ Probing Sweep across ${#STEPS[@]} Checkpoints x ${#SEEDS[@]} Seeds"
echo "=============================================================="

# prepare experiment directory
if [ ! -d "$EXP_PATH" ]; then
  mkdir -p "$EXP_PATH"
fi

# iterate over seeds
for seed in "${SEEDS[@]}"; do
  # prepare data (Pythias have separate tokenizers for seed0 vs others)
  echo "Preprocessing data for ${TASK} task using ${LM_ROOT}-seed${seed}..."
  preprocess "train" "${LM_ROOT}" "${seed}"
  preprocess "dev" "${LM_ROOT}" "${seed}"
  preprocess "test" "${LM_ROOT}" "${seed}"
  # prepare OOD data (if available)
  preprocess "train-ood" "${LM_ROOT}" "${seed}"
  preprocess "dev-ood" "${LM_ROOT}" "${seed}"
  preprocess "test-ood" "${LM_ROOT}" "${seed}"

  # iterate over checkpoint steps
  for step in "${STEPS[@]}"; do
    echo "=========================="
    train $step $seed
    evaluate $step $seed
    (( NUM_EXP++ ))
  done
done

echo "Ran ${NUM_RUN} / ${NUM_EXP} experiments with ${NUM_ERR} error(s)."