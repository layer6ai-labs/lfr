
<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180" height="90"></a>
<a href="https://www.dal.ca///"><img src="https://github.com/layer6ai-labs/lfr/blob/main/logos/dalhousie_university.png" width="180" height="90"></a>
</p>

# Self-supervised Representation Learning from Random Data Projectors

The codebase for [*Self-supervised Representation Learning from Random Data Projectors*](https://arxiv.org/abs/2310.07756).

We proposed a novel self-supervised learning framework by learning from random data projectors (LFR) without any data augmentations. The proposed LFR:

* Do not require domain-specific knowledge or specific model architecture.
* Can be applied on any data modality and application domains.
* Outperforms multiple state-of-the-art SSL baselines on a wide range of data modalities(image, sequential, and tabular) and real-world applications(banking, healthcare and natural sciences).


## Installation & Usage

### Installation
First clone this repository, then navigate to the directory and pip install to install all required packages. 

```
git clone git@github.com:layer6ai-labs/lfr
cd lfr
conda env create -f environment.yml
conda activate lfr
pip install h5py
```

### Usage

```
python main.py --method lfr --dataset kvasir -a resnet18\
        --num_targets 6  --target_layers 2\
        --init-beta --random-dropout\
        --target_sample_ratio 1 --num_of_classes 8\
        --epochs 400 -b 256 --lr 0.0001\
        --optimizer-type sgd --momentum 0.9 --wd 5e-4\
        --fix_pred_lr --train-predictor-individually\
        --pred_epochs 5 --pred_layers 2\
        --eval_epochs 100 --eval_lr 0.001 --eval_bs 256\
        --dim 2048 --pred_dim 256 --loss barlow-batch\
        --num-of-runs 1 
```

See ```scripts/``` for further details on commandline parameters. 


## Data Access

### datasets
MIMIC-III is private dataset. Data pre-processing can be found in this [notebook](notebooks/data-mimic3.ipynb).

All datasets used in our experiments (raw and pre-processed) can be found at [Google drive](https://drive.google.com/drive/folders/1mrl2BLOiDzxOOILcPFOQBfVjnv-13PhS?usp=sharing). Redistribution of these datasets is permitted under their licenses.

To run the code: create ```data/```folder, download and unzip any .zip files into the folder. The original data-preprocessing code for each dataset can be found inside [notebooks folder](notebooks/).

## Citing

If you use any part of this repository in your research, please cite the associated paper with the following bibtex entry:

Authors: Yi Sui, Tongzi Wu, Jesse C. Cresswell, Ga Wu, George Stein, Xiao Shi Huang, Xiaochen Zhang, Maksims Volkovs

```
@misc{sui2023selfsupervised,
      title={Self-supervised Representation Learning from Random Data Projectors}, 
      author={Yi Sui and Tongzi Wu and Jesse C. Cresswell and Ga Wu and George Stein and Xiao Shi Huang and Xiaochen Zhang and Maksims Volkovs},
      year={2023},
      eprint={2310.07756},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This data and code is licensed under the MIT License, copyright by Layer 6 AI.

