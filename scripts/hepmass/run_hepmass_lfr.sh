epochs=20
dim=32
pred_dim=16
lr=1e-6

python main.py --method lfr --dataset hepmass -a mlp \
        --epochs $epochs -b 512 --lr $lr\
        --num_targets $num_targets --fix_pred_lr  \
        --train-predictor-individually  --optimizer-type adam --wd 0\
        --target_layers 2 --pred_epochs 1 --fix_lr\
        --dim $dim --loss barlow-batch  --pred_layers 1\
        --pred_dim $pred_dim --run_eval 1 --num_of_classes 2 --num-of-runs 1 \
        --target_sample_ratio 10 