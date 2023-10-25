#!/bin/bash

# help string
HELP="Usage: run.sh -m model -d dataset [-p pooling] [-t (full|probe)]"

#
# command-line arguments
#
EMBEDDING_POOLING=""  # default embedding pooling is none
TUNING="probe"  # default full fine-tuning is off
while getopts ":f:m:d:p:t:" option; do
   case $option in
      m) # set model
        MODEL=$OPTARG;;
      d) # set data
        TASK=$OPTARG;;
      p) # set embedding pooling
        EMBEDDING_POOLING="--embedding-pooling ${OPTARG} ";;
      t) # set full fine-tuning flag
        TUNING=$OPTARG;;
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

#
# global variables
#
# data variables
DATA_ROOT=~/data/multiberts
RAW_DATA_PATH=$DATA_ROOT/original/$TASK
ALIGNED_DATA_PATH=$DATA_ROOT/aligned/$TASK
if [ "${EMBEDDING_POOLING}" != "" ]; then
  ALIGNED_DATA_PATH=$RAW_DATA_PATH  # switch to non-duplicated labels when pooling is enabled
fi
VALID_SPLIT="dev"
# model setup
LM_ROOT="google/multiberts"
LOCAL_LM_PATH=~/exp/multiberts/pretraining
NUM_LAYERS=13


# experiment setup
EXP_PATH=~/exp/multiberts/"${MODEL//[()\/]}"/$TASK
SEEDS=( 0 1 2 3 4 )
# case: fine-tuning
if [ "${TUNING}" == "full" ]; then
  EXP_PATH=~/exp/multiberts/finetuning/"${MODEL//[()\/]}"/$TASK
  STEPS=(0 1000 5000 20000 2000000)
  EMBEDDING_TUNING="--embedding-tuning "
  NUM_MASKED=$(( "${NUM_LAYERS}" - 1 ))
  LAYER_MASK=$(printf "0,%.0s" $(seq 1 1 $NUM_MASKED))
  LAYER_MASK="${LAYER_MASK}1"
  LAYERS="mask(${LAYER_MASK})"
  BATCH_SIZE=16
  LEARNING_RATE=0.00003
  DECAY_RATE=0.0
  MAX_EPOCHS=30
# case: probing (all checkpoints)
else
  STEPS=(0 10)
  STEPS+=($(seq 100 100 900))
  STEPS+=($(seq 1000 1000 19000))
  STEPS+=($(seq 20000 20000 200000))
  STEPS+=($(seq 300000 100000 2000000))
  EMBEDDING_TUNING=""
  LAYERS="auto(${NUM_LAYERS})"
  BATCH_SIZE=64
  LEARNING_RATE=0.005
  DECAY_RATE=0.0
  MAX_EPOCHS=30
fi
# statistics
NUM_EXP=0
NUM_ERR=0

#
# Helper Functions
#

preprocess() {
  # function arguments
  split=$1
  lm_base=$2
  # check if pre-processed file already exists
  if [ -f "${ALIGNED_DATA_PATH}/${split}.csv" ]; then
    echo "[Warning] Pre-processed data at '${ALIGNED_DATA_PATH}' already exists. Not re-processing."
  else
    if [ -f "${RAW_DATA_PATH}/${split}.csv" ]; then
      # create aligned data directory if not present
      if [ ! -d "${ALIGNED_DATA_PATH}" ]; then
        mkdir -p "${ALIGNED_DATA_PATH}"
      fi
      # apply label repetition according to canonical tokenizer
      python preprocess.py \
        --data-path "${RAW_DATA_PATH}/${split}.csv" \
        --output-path "${ALIGNED_DATA_PATH}/${split}.csv" \
        --model-name "${lm_base}" \
        --repeat-labels
    else
      echo "[Warning] No ${split}-split exists at ${RAW_DATA_PATH}. Skipped."
      return 1
    fi
  fi
}

train() {
  # function arguments
  step=$1
  seed=$2

  # set up language model identifier
  # case: local checkpoint
  if [ "$step" -gt 0 ] && [ "$step" -lt 20000 ]; then
    lm_base="${LM_ROOT}-seed_${seed}-step_0k"
    lm_id="${LM_ROOT#*/}-seed_${seed}-step_${step}"
    encoder_path="${LOCAL_LM_PATH}/seed-${seed}/step-${step}.tar"
    # check if pre-trained encoder exists
    if [ ! -f "${encoder_path}" ]; then
      echo "[Error] Pre-trained encoder at '${encoder_path}' does not exist. Skipping."
      (( NUM_ERR++ ))
      return 1
    fi
    encoder_path="--encoder-path ${encoder_path} "
  # case: remote checkpoint
  else
    lm_base="${LM_ROOT}-seed_${seed}-step_$(( step / 1000 ))k"
    lm_id="${lm_base#*/}"
    encoder_path=""
  fi

  # check if experiment exists
  exp_dir="${EXP_PATH}/${lm_id}"
  if [ -f "$exp_dir/best.pt" ]; then
    echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
  # start training
  else
    echo "Training '${lm_id}' with random seed ${seed}."

    # run training
    python classify.py \
      --train-path $ALIGNED_DATA_PATH/train.csv \
      --valid-path $ALIGNED_DATA_PATH/dev.csv \
      --model-name "${lm_base}" \
      $encoder_path \
      $EMBEDDING_TUNING \
      $EMBEDDING_POOLING \
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
    (( NUM_EXP++ ))
  fi
}

evaluate() {
  # function arguments
  step=$1
  seed=$2

  # set up language model identifier
  # case: local checkpoint
  if [ "$step" -gt 0 ] && [ "$step" -lt 20000 ]; then
    lm_base="${LM_ROOT}-seed_${seed}-step_0k"
    lm_id="${LM_ROOT#*/}-seed_${seed}-step_${step}"
    encoder_path="${LOCAL_LM_PATH}/seed-${seed}/step-${step}.tar"
    # check if pre-trained encoder exists
    if [ ! -f "${encoder_path}" ]; then
      echo "[Error] Pre-trained encoder at '${encoder_path}' does not exist. Skipping."
      (( NUM_ERR++ ))
      return 1
    fi
    encoder_path="--encoder-path ${encoder_path} "
  # case: remote checkpoint
  else
    lm_base="${LM_ROOT}-seed_${seed}-step_$(( step / 1000 ))k"
    lm_id="${lm_base#*/}"
    encoder_path=""
  fi

  # if fully fine-tuning, do not load the initial pre-trained encoder during prediction
  if [ "${TUNING}" == "full" ]; then
    encoder_path=""
  fi

  # check if experiment exists
  exp_dir="${EXP_PATH}/${lm_id}"
  if [ -f "$exp_dir/train-stats.json" ]; then
    echo "[Warning] Training predictions in '$exp_dir' already exist. Not re-predicting."
  # compute final training statistics
  else
    echo "Evaluating '${lm_id}' on $ALIGNED_DATA_PATH/train.csv."
    python classify.py \
      --valid-path $ALIGNED_DATA_PATH/train.csv \
      --model-name "$lm_base" \
      $encoder_path \
      $EMBEDDING_TUNING \
      $EMBEDDING_POOLING \
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
    echo "Evaluating '${lm_id}' on $ALIGNED_DATA_PATH/${VALID_SPLIT}.csv."
    python classify.py \
      --valid-path $ALIGNED_DATA_PATH/${VALID_SPLIT}.csv \
      --model-name "$lm_base" \
      $encoder_path \
      $EMBEDDING_TUNING \
      $EMBEDDING_POOLING \
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
    (( NUM_EXP++ ))
  fi

  # compute and store metrics
  stats_path="${exp_dir}/${VALID_SPLIT}-results.json"
  if [ -f "$stats_path" ]; then
    echo "[Warning] Evaluation results in '$stats_path' already exist. Not re-evaluating."
  else
    # compute evaluation metrics
    python scripts/eval/classification.py \
      --target "$ALIGNED_DATA_PATH/$VALID_SPLIT.csv" \
      --prediction "$exp_dir/$VALID_SPLIT-pred-0.csv" \
      --output "$stats_path"
  fi

  # check if OOD evaluation data is available
  if [ -f "$ALIGNED_DATA_PATH/$VALID_SPLIT-ood.csv" ]; then
    # run OOD prediction
    if [ -f "$exp_dir/${VALID_SPLIT}-ood-pred-0.csv" ]; then
      echo "[Warning] OOD validation predictions in '$exp_dir' already exist. Not re-predicting."
    else
      echo "Evaluating '${lm_id}' on $ALIGNED_DATA_PATH/${VALID_SPLIT}-ood.csv."
      python classify.py \
        --valid-path $ALIGNED_DATA_PATH/${VALID_SPLIT}-ood.csv \
        --model-name "$lm_base" \
        $encoder_path \
        $EMBEDDING_TUNING \
        $EMBEDDING_POOLING \
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
      (( NUM_EXP++ ))
    fi
    # compute and store metrics
    if [ -f "${exp_dir}/${VALID_SPLIT}-ood-results.json" ]; then
      echo "[Warning] Evaluation results in '$stats_path' already exist. Not re-evaluating."
    else
      # compute evaluation metrics
      python scripts/eval/classification.py \
        --target "$ALIGNED_DATA_PATH/$VALID_SPLIT-ood.csv" \
        --prediction "$exp_dir/$VALID_SPLIT-ood-pred-0.csv" \
        --output "${exp_dir}/${VALID_SPLIT}-ood-results.json"
    fi
  fi
}

#
# Main
#

echo "===================================================="
echo "Sweep across ${#STEPS[@]} Checkpoints x ${#SEEDS[@]} Seeds"
if [ "${TUNING}" == "full" ]; then
  echo "🔥 Full-fine Tuning"
else
  echo "❄️ Probing"
fi
echo "===================================================="

# prepare data (MultiBERTs share encoder)
echo "Preprocessing data for ${TASK} task using ${LM_ROOT}..."
preprocess "train" "${LM_ROOT}-seed_0-step_0k"
preprocess "dev" "${LM_ROOT}-seed_0-step_0k"
preprocess "test" "${LM_ROOT}-seed_0-step_0k"
# prepare OOD data (if available)
preprocess "train-ood" "${LM_ROOT}-seed_0-step_0k"
preprocess "dev-ood" "${LM_ROOT}-seed_0-step_0k"
preprocess "test-ood" "${LM_ROOT}-seed_0-step_0k"

# prepare experiment directory
if [ ! -d "$EXP_PATH" ]; then
  mkdir -p "$EXP_PATH"
fi

# iterate over seeds
for seed in "${SEEDS[@]}"; do
  for step in "${STEPS[@]}"; do
    echo "=========================="
    train $step $seed
    evaluate $step $seed
  done
done

echo "Completed ${NUM_EXP} experiments with ${NUM_ERR} error(s)."