#!/usr/bin/env bash
epochs=200 
eval_epochs=100 

dim=2304
pred_dim=256

python main.py --method autoencoder --dataset har -a cnn\
        --num_targets 0  --target_layers 1\
        --epochs $epochs -b 128 --lr 3e-4\
        --eval_lr 3e-4 --eval_bs 128 --eval_epochs $eval_epochs --eval_wd 3e-4\
        --optimizer-type adam --wd 3e-4 --fix_lr --fix_eval_lr \
        --train-predictor-individually --pred_epochs 5 --pred_layers 1\
        --dim $dim --pred_dim $pred_dim --loss mse --num_of_classes 6\
        --num-of-runs 1
