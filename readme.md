# Tx Metalearning

The aim of this project is to build on the good work in [Smith et al.](<https://www.biorxiv.org/content/10.1101/574723v2>)

* Try better machine learning classifiers
* Try feature selection methods
* Try better embedding methods
* Try meta-learning

## To do

1. Duplicate benchmarks from [Smith et al.](<https://www.biorxiv.org/content/10.1101/574723v2>) for knn and L2
2. Implement E0 / E632
3. L2 with feature selection. Rather than learning a low-level representation, what about using INVASE (e.g. feature selection)
4. Try Neural Tangent Kernels as an additional baseline
5. NODE??
6. The Pfizer paper used VAEs, AEs etc. Are new methods better (e.g. beta-VAEs)?  Can we disentangle bias? <https://arxiv.org/abs/1905.05300v1>
7. Flow model - the nonlinear ICA one?
8. Meta-learning - Warped GD

## Notes



## Code
```bash
git clone git@github.com:notreallyme2/tx-metalearning.git
git clone https://github.com/unlearnai/representation_learning_for_transcriptomics.git
git clone git@github.com:notreallyme2/torch-templates.git
```

## Data 

To download the data:

```bash
mkdir ~/data/pfizer_tx
cd ~/data/pfizer_tx

wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/14565026/README.md
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/14565029/CCBY4.0license.txt

# supervised data
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/14548094/tasks_README.md # readme
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/14546420/tasks_all_clr.tar.gz # data

# unzip
tar -zxvf tasks_all_clr.tar.gz

# unsupervised data
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/14548157/unsupervised_README.md # readme
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/14553134/unsupervised_all_clr_train.h5 # data
```
