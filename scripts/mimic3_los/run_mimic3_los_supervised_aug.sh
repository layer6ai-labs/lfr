#!/usr/bin/env bash
epochs=10 
bs=4096 
lr=5e-6 
ckpt_path=ckpt

dim=10
pred_dim=256


python main.py --method supervised-aug --dataset mimic3-los -a tcn\
        --num_targets 0 --pred_layers 2 --target_layers 1\
        --dim $dim --pred_dim $pred_dim --loss ce\
         --epochs $epochs -b $bs --lr $lr\
        --optimizer-type adam --momentum 0.9 --wd 5e-4\
        --num-of-runs 1 --num_of_classes 10