#!/bin/bash
# Batch 2: StackCube (Medium) + TwoRobotPickCube — All Baselines, 5 Seeds, 30 Epochs
# Runs 2 tasks in parallel (one per GPU)

PYTHON=/home/devashri/miniconda3/envs/sptom_icml/bin/python
SCRIPT=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/scripts/train.py
DATA=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/data/maniskill
RESULTS=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/results/batch2

EPOCHS=30
SEEDS=5
BATCH=16

mkdir -p $RESULTS

echo "============================================================"
echo "BATCH 2: StackCube + TwoRobotPickCube - Started $(date)"
echo "============================================================"

# --- StackCube (GPU 0) + TwoRobotPickCube (GPU 1) ---
echo ""
echo "--- StackCube (GPU 0) + TwoRobotPickCube (GPU 1) ---"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  --data_path $DATA/stackcube_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/stackcube \
  > $RESULTS/stackcube.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  --data_path $DATA/tworobotpickcube_v1_data.npz \
  --all_baselines --epochs $EPOCHS --seeds $SEEDS --batch_size $BATCH \
  --output_dir $RESULTS/tworobotpickcube \
  > $RESULTS/tworobotpickcube.log 2>&1 &
PID2=$!

echo "  StackCube PID=$PID1 (GPU 0), TwoRobotPickCube PID=$PID2 (GPU 1)"
wait $PID1 $PID2
echo "  Done: $(date)"

echo ""
echo "============================================================"
echo "BATCH 2 COMPLETE: $(date)"
echo "============================================================"
echo "Results:"
for task in stackcube tworobotpickcube; do
  if [ -f "$RESULTS/$task/results.json" ]; then
    echo "  $task: OK"
  else
    echo "  $task: FAILED (check $RESULTS/$task.log)"
  fi
done
