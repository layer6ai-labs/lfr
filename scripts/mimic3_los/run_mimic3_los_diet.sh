#!/usr/bin/env bash
epochs=2000
eval_epochs=300
bs=512
lr=5e-4
eval_lr=5e-5
dim=64
pred_dim=256

python main.py --method diet --dataset mimic3-los -a tcn\
        --num_targets 0  --pred_layers 2 --target_layers 1\
        --num_of_classes 10 --warmup_epochs 10\
        --dim $dim --pred_dim $pred_dim --loss ce-smooth\
         --epochs $epochs -b $bs --lr $lr\
        --optimizer-type adam --momentum 0.9 --wd 5e-4\
        --eval_epochs $eval_epochs --eval_lr $eval_lr --eval_bs $bs\
        --num-of-runs 1 --train_with_steps