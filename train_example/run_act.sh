bs=8
ws=2
torchrun --standalone --nnodes=1 --nproc-per-node=2 train_ACT_example.py \
    --dataset 'airkitchen' \
    --algo_name 'act' \
    --ddp True \
    --img_size 0 \
    --visual_encoder resnet18 \
    --visual_pretrain True \
    --ft_vision True \
    --ac_num 4 \
    --num_encoder_layers 4 \
    --num_decoder_layers 7 \
    --norm minmax \
    --discretize_actions False \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.00001 \
    --val_freq 20000 \
    --resume None \
    --wandb True \
    --steps 400000 \
    --save True \
    --save_freq 50000 \
    --save_path ../experiments/airkitchen/ACT/40W_bs8_noqpos \
    --log_path ../experiments/airkitchen/ACT/40W_bs8_noqpos \
    --port 2051 \
