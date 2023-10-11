#!/usr/bin/env bash
epochs=100 
dim=256
pred_dim=256

lr=0.001
python main.py --method stab --dataset theorem -a mlp\
        --num_targets 0  --target_layers --proj_layers 2 --proj_dim 256\
        --epochs $epochs -b 128 --lr $lr\
        --optimizer-type adam --wd 0 --fix_lr --fix_eval_lr\
        --fix_pred_lr --pred_layers 1 --num_of_classes 6\
        --dim $dim --pred_dim $pred_dim --loss cosine\
        --num-of-runs 1 --stab-drop-rate 0.8