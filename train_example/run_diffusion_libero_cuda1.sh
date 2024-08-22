export MUJOCO_GL="osmesa"
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=cdad4c82816c6da591e611ae4ac5a07ee7f4611a

bs=64
ws=2
text_encoder=CLIP
file_name=libero130_llm_lang_clip_r50_frozen_7800ep_0822

for seed in 42; do
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_diffusion_policy_example_libero.py \
    --dataset_name 'libero130' \
    --algo_name 'diffusion visual motor' \
    --ddp False \
    --img_size 128 \
    --seed $seed \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --ft_vision True \
    --film_fusion False \
    --ac_num 6 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 9 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0003 \
    --val_freq 10000000 \
    --eval_freq 200000 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 200000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --log_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --port 2077 \
    --add_noise False \
    --minus_mean False \
    --mean_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/mean_ood_7800ep.npz \
    --noise_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/noise_DecisionNCE-T_all_endbegin_10800ep.npz \
    --lang_prop '' \
    --json_copy 10 \
    --cos_noise 0.2 \
    --cos_noise_decay 0.8 
done

wait

text_encoder=T5
file_name=libero130_llm_lang_t5_r50_frozen_7800ep_0822

for seed in 42; do
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_diffusion_policy_example_libero.py \
    --dataset_name 'libero130' \
    --algo_name 'diffusion visual motor' \
    --ddp False \
    --img_size 128 \
    --seed $seed \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --ft_vision True \
    --film_fusion False \
    --ac_num 6 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 9 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0003 \
    --val_freq 10000000 \
    --eval_freq 200000 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 200000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --log_path ../experiments/libero/libero_goal/diffusion/$file_name \
    --port 2077 \
    --add_noise False \
    --minus_mean False \
    --mean_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/mean_ood_7800ep.npz \
    --noise_data_path /home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/noise_DecisionNCE-T_all_endbegin_10800ep.npz \
    --lang_prop '' \
    --json_copy 10 \
    --cos_noise 0.2 \
    --cos_noise_decay 0.8 
done

wait


./run_incase_cuda1.sh