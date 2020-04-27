# Tx Metalearning

## To do

1. Download data to AWS
2. Duplicate benchmarks from Pfizer paper
3. 

## Notes
* We followed the methods in [pfizer paper](papers/pfizer-tx.pdf)
* Following the recommendations in xxx, we used the full gene expression datasets, CLR normalized and downloaded from xxx.
* We focused on the 24 classification tasks

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
