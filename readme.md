# Tx Metalearning

## To do

1. Download data to AWS - DONE
2. Duplicate benchmarks from Pfizer paper
3. Try Neural Tangent Kernels as an additional baseline
4. Better VAEs - disentangled? 
5. Flow model - the nonlinear ICA one?
6. Meta-learning - Warped GD

## Requirements
'pip install pytorch-lightning'

## Notes
* We followed the methods in [pfizer paper](papers/pfizer-tx.pdf)
* Following the recommendations in xxx, we used the full gene expression datasets, CLR normalized and downloaded from xxx.
* We focused on the 24 classification tasks
* Two papers? Meta-learning and causal/better AEs

### On nested cross-validation

The aim of nested cross-validation is *not to select a model with great hyperparameters*.  
Rather, it is to test our overall modelling methodology.  
*Can our way of choosing a good set of hyperparameters be expected to find a good final model?*

From <https://weina.me/nested-cross-validation>: "...if the model is stable (does not change much if the training data is perturbed), the hyperparameter found in each outer loop may be the same (using grid search) or similar to each other (using random search)."

This is the best way to compare each method.

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
