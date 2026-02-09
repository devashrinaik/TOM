#!/bin/bash
# Phase 1: Easy Tasks - All Baselines, 5 Seeds, 30 Epochs
# Runs 2 tasks in parallel (one per GPU) — needs ~18GB RAM each

PYTHON=/home/devashri/miniconda3/envs/sptom_icml/bin/python
SCRIPT=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/scripts/train.py
DATA=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/data/maniskill
RESULTS=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/results/phase1_easy

EPOCHS=30
SEEDS=5
BATCH=16

echo "============================================================"
echo "PHASE 1: Easy Tasks (Parallel) - Started $(date)"
echo "============================================================"

# --- Pair 1: PickCube (GPU 0) + PushCube (GPU 1) ---
echo ""
echo "--- Pair 1/2: PickCube (GPU 0) + PushCube (GPU 1) ---"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  --data_path $DATA/pickcube_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/pickcube \
  > $RESULTS/pickcube.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  --data_path $DATA/pushcube_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/pushcube \
  > $RESULTS/pushcube.log 2>&1 &
PID2=$!

echo "  PickCube PID=$PID1 (GPU 0), PushCube PID=$PID2 (GPU 1)"
wait $PID1 $PID2
echo "  Pair 1 done: $(date)"

# --- Pair 2: PokeCube (GPU 0) + PullCube (GPU 1) ---
echo ""
echo "--- Pair 2/2: PokeCube (GPU 0) + PullCube (GPU 1) ---"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  --data_path $DATA/pokecube_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/pokecube \
  > $RESULTS/pokecube.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  --data_path $DATA/pullcube_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/pullcube \
  > $RESULTS/pullcube.log 2>&1 &
PID4=$!

echo "  PokeCube PID=$PID3 (GPU 0), PullCube PID=$PID4 (GPU 1)"
wait $PID3 $PID4
echo "  Pair 2 done: $(date)"

echo ""
echo "============================================================"
echo "PHASE 1 COMPLETE: $(date)"
echo "============================================================"
echo "Results:"
for task in pickcube pushcube pokecube pullcube; do
  if [ -f "$RESULTS/$task/results.json" ]; then
    echo "  $task: OK"
  else
    echo "  $task: FAILED (check $RESULTS/$task.log)"
  fi
done
