epochs=20
dim=32
pred_dim=16
lr=1e-6

python main.py --method autoencoder --dataset hepmass -a mlp \
        --epochs $epochs -b 512 --lr $lr \
        --fix_pred_lr   --num_targets 0\
        --optimizer-type adam --wd 0 --fix_lr\
        --dim $dim --loss mse --pred_layers 1\
        --pred_dim $pred_dim --num_of_classes 2 --num-of-runs 1 --fix_lr 