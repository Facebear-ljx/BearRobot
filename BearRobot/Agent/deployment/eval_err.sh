#!/bin/bash

file_name=trial_err_09

export MUJOCO_GL="osmesa"
export CUDA_VISIBLE_DEVICES=0
python Err_eval.py\
    --k 0.2 \
    --ckpt_path "/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/experiments/libero/libero_goal/$file_name/latest.pth"\
    --statistic_path "/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/experiments/libero/libero_goal/$file_name/statistics.json"\
    --num_episodes 5 \
    --eval_horizon 200 



