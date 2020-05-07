# Tx Metalearning

The aim of this project is to build on the good work in [Smith et al.](<https://www.biorxiv.org/content/10.1101/574723v2>)

* Try better machine learning classifiers
* Try feature selection methods
* Try better embedding methods
* Try meta-learning

## To do

1. Duplicate benchmarks from [Smith et al.](<https://www.biorxiv.org/content/10.1101/574723v2>)
2. Implement E0 / E632
3. Try Neural Tangent Kernels as an additional baseline
4. NODE??
5. Better VAEs - disentangled? 
6. Flow model - the nonlinear ICA one?
7. Meta-learning - Warped GD

## Notes
* We followed the methods in [pfizer paper](papers/pfizer-tx.pdf)
* Following the recommendations in xxx, we used the full gene expression datasets, CLR normalized and downloaded from xxx.
* We focused on the 24 classification tasks
* Two papers? Meta-learning and causal/better AEs

If I choose to do survival analysis later, [this](<http://www.sthda.com/english/wiki/survival-analysis-basics>) is a really good quick primer.

### On nested cross-validation

First important point: if you use CV to select your best hyperparameters, the performance measures from this process are over-estimates of generalised performance (because you are picking the best result).
So... you need to hold out a third data set (your test set, d'uh!!).
To be more robust - to reduce the impact of sampling error on your final performance estimate - you do nested CV.

The aim of nested cross-validation is *not to select a model with great hyperparameters*.  
Rather, it is to test our overall modelling methodology.  
*Can our way of choosing a good set of hyperparameters be expected to find a good final model?*

From <https://weina.me/nested-cross-validation>: "...if the model is stable (does not change much if the training data is perturbed), the hyperparameter found in each outer loop may be the same (using grid search) or similar to each other (using random search)."

This is the best way to compare each method.
Are there less computationally expensive alternatives?

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
