export MUJOCO_GL="osmesa"
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=cdad4c82816c6da591e611ae4ac5a07ee7f4611a

bs=64
ws=2
text_encoder=DecisionNCE-V
file_name=airkitchen_lang_r50_frozen_7800ep_noise_0.2_0821

for seed in 42; do
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_diffusion_policy_example.py \
    --dataset_name 'airkitchen_lang' \
    --algo_name 'diffusion visual motor' \
    --ddp False \
    --img_size 224 \
    --seed $seed \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --ft_vision True \
    --film_fusion False \
    --ac_num 4 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 0 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0001 \
    --val_freq 100000000000 \
    --eval_freq 100000000000 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 100000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --log_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --port 2077 \
    --add_noise True \
    --minus_mean True \
    --mean_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/mean_ood_7800ep.npz \
    --noise_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/noise_DecisionNCE-T_all_endbegin_10800ep.npz \
    --lang_prop '' \
    --json_copy 1 \
    --cos_noise 0.2 \
    --cos_noise_decay 0.8 
done

wait

file_name=airkitchen_lang_r50_frozen_7800ep_noise_0.1_0821

for seed in 42; do
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_diffusion_policy_example.py \
    --dataset_name 'airkitchen_lang' \
    --algo_name 'diffusion visual motor' \
    --ddp False \
    --img_size 224 \
    --seed $seed \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --ft_vision True \
    --film_fusion False \
    --ac_num 4 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 0 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0001 \
    --val_freq 100000000000 \
    --eval_freq 100000000000 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 100000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --log_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --port 2077 \
    --add_noise True \
    --minus_mean True \
    --mean_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/mean_ood_7800ep.npz \
    --noise_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/noise_DecisionNCE-T_all_endbegin_10800ep.npz \
    --lang_prop '' \
    --json_copy 1 \
    --cos_noise 0.1 \
    --cos_noise_decay 0.9 
done

wait

file_name=airkitchen_lang_r50_frozen_7800ep_noise_0.3_0821

for seed in 42; do
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_diffusion_policy_example.py \
    --dataset_name 'airkitchen_lang' \
    --algo_name 'diffusion visual motor' \
    --ddp False \
    --img_size 224 \
    --seed $seed \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --ft_vision True \
    --film_fusion False \
    --ac_num 4 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 0 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0001 \
    --val_freq 100000000000 \
    --eval_freq 100000000000 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 100000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --log_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --port 2077 \
    --add_noise True \
    --minus_mean True \
    --mean_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/mean_ood_7800ep.npz \
    --noise_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/noise_DecisionNCE-T_all_endbegin_10800ep.npz \
    --lang_prop '' \
    --json_copy 1 \
    --cos_noise 0.3 \
    --cos_noise_decay 0.7 
done

wait

./run_incase_cuda1.sh