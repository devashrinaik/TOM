#!/bin/bash
# Recovery Curve Analysis for Batch 2 tasks
# Runs recovery analysis on StackCube and TwoRobotPickCube

PYTHON=/home/devashri/miniconda3/envs/sptom_icml/bin/python
SCRIPT=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/scripts/recovery_analysis.py
DATA=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/data/maniskill
RESULTS=/mnt/ssd1/devashri/TOM_revision/ua_tom_2/results/recovery

mkdir -p $RESULTS

echo "============================================================"
echo "Recovery Curve Analysis - Started $(date)"
echo "============================================================"

# StackCube on GPU 0
echo ""
echo "--- StackCube (GPU 0) ---"
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  --data_path $DATA/stackcube_v1_data.npz \
  --output_dir $RESULTS/stackcube \
  --models ua_tom mamba transformer gru tomnet bocpd context_cond liam \
  --epochs 30 --seed 0 --max_offset 20 \
  > $RESULTS/stackcube.log 2>&1 &
PID1=$!

# TwoRobotPickCube on GPU 1
echo "--- TwoRobotPickCube (GPU 1) ---"
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  --data_path $DATA/tworobotpickcube_v1_data.npz \
  --output_dir $RESULTS/tworobotpickcube \
  --models ua_tom mamba transformer gru tomnet bocpd context_cond liam \
  --epochs 30 --seed 0 --max_offset 20 \
  > $RESULTS/tworobotpickcube.log 2>&1 &
PID2=$!

echo "  StackCube PID=$PID1 (GPU 0), TwoRobotPickCube PID=$PID2 (GPU 1)"
wait $PID1 $PID2
echo ""
echo "============================================================"
echo "Recovery Analysis COMPLETE: $(date)"
echo "============================================================"
