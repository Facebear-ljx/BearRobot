bs=64
ws=2
export WANDB_API_KEY=265854b7b5037e14094d66ddd65204191693217d
torchrun --standalone --nnodes=1 --nproc-per-node=2 train_ACT_airbot_example.py \
    --dataset_name 'airbot' \
    --algo_name 'act' \
    --ddp True \
    --img_size 224 \
    --visual_encoder resnet18 \
    --visual_pretrain True \
    --ft_vision True \
    --ac_num 4 \
    --s_dim 7 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --num_encoder_layers 4 \
    --num_decoder_layers 7 \
    --norm mean \
    --discretize_actions False \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.00003 \
    --val_freq 200000000 \
    --eval_freq 200000000 \
    --resume None \
    --wandb True \
    --steps 600000 \
    --save True \
    --save_freq 25000 \
    --save_path ../experiments/airbot/ACT/1203 \
    --log_path ../experiments/airbot/ACT/1203 \
    --port 2051 \
