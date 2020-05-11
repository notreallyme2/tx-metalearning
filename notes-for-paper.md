# Notes for the paper(s)

* We followed the methods in [pfizer paper](papers/pfizer-tx.pdf)
* Following the recommendations in xxx, we used the full gene expression datasets, CLR normalized and downloaded from xxx.
* We focused on the 24 classification tasks
* Two papers? Meta-learning and causal/better AEs

* Smith et al.'s recommendations seem overly conservative.
  * When does RF outperform LR? Is it related to $n$?
  * Is their methodology penalizing more complex models (each model is built on 4/5s of the data). What about using repeated CV for model selection and E0 for model assessment?


Having read Krstajic et al., I'm inclined to use Repeated CV rather than Nested.
It uses more of the data.
It has much lower variance most of the time.
We can use something like E0 to assess model performance on a few selected models.


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
