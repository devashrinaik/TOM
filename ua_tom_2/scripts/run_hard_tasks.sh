#!/bin/bash
# Hard Tasks: PegInsertionSide + TwoRobotStackCube — All Baselines, 5 Seeds, 30 Epochs
# Runs 2 tasks in parallel (one per GPU)

PYTHON=/home/devashri/miniconda3/envs/sptom_icml/bin/python
SCRIPT=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/scripts/train.py
DATA=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/data/maniskill
RESULTS=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/results/hard_tasks

EPOCHS=30
SEEDS=5
BATCH=16

mkdir -p $RESULTS

echo "============================================================"
echo "HARD TASKS: PegInsertionSide + TwoRobotStackCube - Started $(date)"
echo "============================================================"

# --- PegInsertionSide (GPU 0) + TwoRobotStackCube (GPU 1) ---
echo ""
echo "--- PegInsertionSide (GPU 0) + TwoRobotStackCube (GPU 1) ---"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  --data_path $DATA/peginsertionside_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/peginsertionside \
  > $RESULTS/peginsertionside.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  --data_path $DATA/tworobotstackcube_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/tworobotstackcube \
  > $RESULTS/tworobotstackcube.log 2>&1 &
PID2=$!

echo "  PegInsertionSide PID=$PID1 (GPU 0), TwoRobotStackCube PID=$PID2 (GPU 1)"
wait $PID1 $PID2
echo "  Done: $(date)"

echo ""
echo "============================================================"
echo "HARD TASKS COMPLETE: $(date)"
echo "============================================================"
echo "Results:"
for task in peginsertionside tworobotstackcube; do
  if [ -f "$RESULTS/$task/results.json" ]; then
    echo "  $task: OK"
  else
    echo "  $task: FAILED (check $RESULTS/$task.log)"
  fi
done
