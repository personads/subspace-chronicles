#!/bin/bash

DATE=$(echo -n `date '+%y%m%d'`)
MODEL="70m"
LAYERS="7"

bash scripts/pythia/run.sh -m "mdl/linear()" -d "pos" -t "on" -s "${MODEL}" -l "${LAYERS}" 2>&1 | tee -a logs/$DATE-pythia-${MODEL}.log
bash scripts/pythia/run.sh -m "mdl/linear()" -d "dep" -t "on" -s "${MODEL}" -l "${LAYERS}" 2>&1 | tee -a logs/$DATE-pythia-${MODEL}.log
bash scripts/pythia/run.sh -m "mdl/linear()" -d "semtag" -t "on" -s "${MODEL}" -l "${LAYERS}" 2>&1 | tee -a logs/$DATE-pythia-${MODEL}.log
bash scripts/pythia/run.sh -m "mdl/linear()" -d "coref" -t "on" -s "${MODEL}" -l "${LAYERS}" 2>&1 | tee -a logs/$DATE-pythia-${MODEL}.log
bash scripts/pythia/run.sh -m "mdl/linear()" -d "ner" -t "on" -s "${MODEL}" -l "${LAYERS}" 2>&1 | tee -a logs/$DATE-pythia-${MODEL}.log
bash scripts/pythia/run.sh -m "mdl/linear()" -d "senti" -s "${MODEL}" -l "${LAYERS}" 2>&1 | tee -a logs/$DATE-pythia-${MODEL}.log
bash scripts/pythia/run.sh -m "mdl/linear()" -d "topic" -s "${MODEL}" -l "${LAYERS}" 2>&1 | tee -a logs/$DATE-pythia-${MODEL}.log