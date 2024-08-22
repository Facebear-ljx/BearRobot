export CUDA_VISIBLE_DEVICES=1

basic_path=/home/dodo/.zh1hao_space/bear_branch/BearRobot/experiments/libero/libero_goal/diffusion/libero130_lang_1.0_CLIP_noise_0.5_0816/
task_suite_name=libero_goal
json_path=/home/dodo/ljx/BearRobot/data/libero/$task_suite_name-ac.json
ckpt_name=20w.pth

# Output debugging information
echo "Basic path: $basic_path"
echo "Checkpoint name: $ckpt_name"
echo "Task suite name: $task_suite_name"
echo "Json path: $json_path"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"


# Run the evaluation script
python eval_libero.py \
    --basic_path $basic_path \
    --statistic_name statistics.json \
    --ckpt_name $ckpt_name \
    --save_path $basic_path \
    --json_path $json_path\
    --task_suite_name $task_suite_name \
    --num_episodes 10 \
    --eval_horizon 300 \
    --cross_modal True 

wait


basic_path=/home/dodo/.zh1hao_space/bear_branch/BearRobot/experiments/libero/libero_goal/diffusion/libero130_lang_1.0_CLIP_noise_0.4_0816/
task_suite_name=libero_goal
json_path=/home/dodo/ljx/BearRobot/data/libero/$task_suite_name-ac.json
ckpt_name=20w.pth

# Output debugging information
echo "Basic path: $basic_path"
echo "Checkpoint name: $ckpt_name"
echo "Task suite name: $task_suite_name"
echo "Json path: $json_path"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"


# Run the evaluation script
python eval_libero.py \
    --basic_path $basic_path \
    --statistic_name statistics.json \
    --ckpt_name $ckpt_name \
    --save_path $basic_path \
    --json_path $json_path\
    --task_suite_name $task_suite_name \
    --num_episodes 10 \
    --eval_horizon 300 \
    --cross_modal True 

wait

basic_path=/home/dodo/.zh1hao_space/bear_branch/BearRobot/experiments/libero/libero_goal/diffusion/libero130_lang_1.0_CLIP_noise_0.3_0816/
task_suite_name=libero_goal
json_path=/home/dodo/ljx/BearRobot/data/libero/$task_suite_name-ac.json
ckpt_name=20w.pth

# Output debugging information
echo "Basic path: $basic_path"
echo "Checkpoint name: $ckpt_name"
echo "Task suite name: $task_suite_name"
echo "Json path: $json_path"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"


# Run the evaluation script
python eval_libero.py \
    --basic_path $basic_path \
    --statistic_name statistics.json \
    --ckpt_name $ckpt_name \
    --save_path $basic_path \
    --json_path $json_path\
    --task_suite_name $task_suite_name \
    --num_episodes 10 \
    --eval_horizon 300 \
    --cross_modal True 

wait

basic_path=/home/dodo/.zh1hao_space/bear_branch/BearRobot/experiments/libero/libero_goal/diffusion/libero130_lang_1.0_CLIP_noise_0.2_0816/
task_suite_name=libero_goal
json_path=/home/dodo/ljx/BearRobot/data/libero/$task_suite_name-ac.json
ckpt_name=20w.pth

# Output debugging information
echo "Basic path: $basic_path"
echo "Checkpoint name: $ckpt_name"
echo "Task suite name: $task_suite_name"
echo "Json path: $json_path"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"


# Run the evaluation script
python eval_libero.py \
    --basic_path $basic_path \
    --statistic_name statistics.json \
    --ckpt_name $ckpt_name \
    --save_path $basic_path \
    --json_path $json_path\
    --task_suite_name $task_suite_name \
    --num_episodes 10 \
    --eval_horizon 300 \
    --cross_modal True 

wait

basic_path=/home/dodo/.zh1hao_space/bear_branch/BearRobot/experiments/libero/libero_goal/diffusion/libero130_lang_1.0_CLIP_noise_0.1_0816/
task_suite_name=libero_goal
json_path=/home/dodo/ljx/BearRobot/data/libero/$task_suite_name-ac.json
ckpt_name=20w.pth

# Output debugging information
echo "Basic path: $basic_path"
echo "Checkpoint name: $ckpt_name"
echo "Task suite name: $task_suite_name"
echo "Json path: $json_path"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"


# Run the evaluation script
python eval_libero.py \
    --basic_path $basic_path \
    --statistic_name statistics.json \
    --ckpt_name $ckpt_name \
    --save_path $basic_path \
    --json_path $json_path\
    --task_suite_name $task_suite_name \
    --num_episodes 10 \
    --eval_horizon 300 \
    --cross_modal True 

wait

./eval_libero_incase.sh
