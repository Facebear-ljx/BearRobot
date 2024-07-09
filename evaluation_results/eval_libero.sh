#!/bin/bash

# Optionally activate conda environment
# Uncomment the lines below if needed
# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate myenv

basic_path=/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion/resnet34_wstate_0706_ff_all600ep_DecisionNCE-V/

# Output debugging information
echo "Basic path: $basic_path"
echo "Python version: $(python --version)"
echo "Conda environment: $(conda info --envs)"

# Check if required files exist
if [ ! -f "${basic_path}latest.pth" ]; then
  echo "Checkpoint file not found!"
  exit 1
fi

if [ ! -f "${basic_path}statistics.json" ]; then
  echo "Statistics file not found!"
  exit 1
fi

if [ ! -f "/home/dodo/ljx/BearRobot/data/libero/libero_goal-ac.json" ]; then
  echo "JSON file not found!"
  exit 1
fi

# Run the evaluation script
python eval_libero.py \
    --basic_path $basic_path \
    --statistic_name statistics.json \
    --ckpt_name latest.pth \
    --save_path $basic_path \
    --json_path /home/dodo/ljx/BearRobot/data/libero/libero_goal-ac.json \
    --task_suite_name libero_goal \
    --num_episodes 10 \
    --eval_horizon 300 \
    --cross_modal False > eval_output.log 2>&1

# Check exit status
if [ $? -ne 0 ]; then
  echo "Evaluation script failed!"
  exit 1
fi

echo "Evaluation completed successfully."
