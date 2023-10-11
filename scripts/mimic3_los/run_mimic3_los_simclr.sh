#!/usr/bin/env bash
epochs=600 
bs=4096 
lr=1e-3 
warmup_lr=1e-5 
warmup_epochs=60
ckpt_path=ckpt
eval_lr=1e-4 
eval_epochs=300

dim=64
pred_dim=256


python main.py --method simclr --dataset mimic3-los -a tcn \
         --epochs $epochs --train_with_steps -b $bs --optimizer-type adam --momentum 0.9 --wd 5e-4 \
        --lr $lr --warmup_lr $warmup_lr --warmup_epochs $warmup_epochs \
        --num_targets 0 --target_layers 1 \
        --eval_epochs $eval_epochs --eval_lr $eval_lr --fix_eval_lr \
        --num_of_classes 10 \
        --dim $dim --loss ce --pred_layers 2 --eval_bs $bs \
        --pred_dim $pred_dim --num-of-runs 1