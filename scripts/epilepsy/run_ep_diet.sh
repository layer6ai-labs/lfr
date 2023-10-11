#!/usr/bin/env bash
epochs=500 
eval_epochs=100 

dim=3072
pred_dim=256

python main.py --method diet --dataset epilepsy -a cnn\
        --num_targets 0  --target_layers 1 --num_of_classes 2\
        --epochs $epochs -b 128 --lr 3e-4\
        --eval_lr 3e-4 --eval_bs 128 --eval_epochs $eval_epochs --eval_wd 3e-4\
        --optimizer-type adam --wd 3e-4 --fix_lr --fix_eval_lr \
        --dim $dim --pred_dim $pred_dim --loss ce-smooth\
        --num-of-runs 1

   


