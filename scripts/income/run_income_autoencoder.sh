epochs=100
dim=256
pred_dim=256

lr=0.001


python main.py --method autoencoder --dataset uci-income -a mlp \
        --epochs $epochs -b 128 --lr $lr \
        --fix_pred_lr --num_targets 0\
        --optimizer-type adam --wd 0 --fix_lr\
        --dim $dim --loss mse --pred_layers 1\
        --pred_dim $pred_dim --num_of_classes 2 --num-of-runs 1\
