## This is my attempt to beat [Herbarium 2020 FGVC7 competition](https://www.kaggle.com/c/herbarium-2020-fgvc7)


### 0. Dependencies:
* `pytorch==1.5.0`
* `pretrainedmodels==0.7.4`
* `pytorch-lightning==0.7.1`

### 1. Code included:
* `dataset.py` works with Herbarium dataset 
* `losses.py` with `FocalLoss` implementation
* `model.py` with model and logic
* `utils.py` with slightly modified functions from `pretrainedmodels` package
* `collect_stats.py` script that collects mean and std over all images

### 2. Model description
I've tried to squeeze from `se_resnext50_32x4d` feature generator as much as possible. The main source of ideas was this [A Bag of Freebies video](https://www.youtube.com/watch?v=iRrDwzcxg08). 
There were 32k of classes that hierarchically arranged in superclasses with smaller number of distinct labels. In my model I've pursued them all with different weights before loss.
The loss was a weighted FocalLoss from this [paper](https://arxiv.org/pdf/1901.05555.pdf). I've analyzed in detail how much that improved performance, but it seems obvious that one should try to weight such unbalanced data. As info says:
> Each category has at least 1 instance in both the training and test datasets. Note that the test set distribution is slightly different from the training set distribution. The training set contains species with hundreds of examples, but the test set has the number of examples per species capped at a maximum of 10. 


I've collected statistics of mean and variance of whole dataset to normalize input images and that has given a little boost. 
The next improvement I've introduced was decaying optimizer learning rate. It starts from `30e-5` and divided by two each time every `5` or `10` steps. I've used Adam optimizator. There was an attempt to switch oprimizators during training from Adam to SGD and vice versa. I called this paradigm AdamEva optimizer. The attempt crashed against the constraints of `pytorch-lightning` framework that allow you to use several optimizers but each optimizer has it's own `train` call. 

Finally, about augmentations and TTA. Random crop, random flip of herbarium image improves performance but not a Color Jitter, that's my insight. For the TTA I've sampled 4 inputs. I've done 3 times train augmentation than 1 more time just normalization and resize. After that logits of these four samples have averaged to produce output logits.

## 3. What also hasn't work

1. Metric Learning

I've not figured out how to do metric learning for such a big dataset with such amount of unique labels. I've engineered how to store this amount of data but haven't engineered a way to use it

2. Descriminator Network

The main idea is to train a network to predict whether image from train or test set. This might help to create robust validation set as a part of train set. Good validation set gives you an ability to know how good your machine learning model is. Sounds obvious, but it isn't easy to find such when your dataset very disbalanced. By the way, I've used a random subset of a train as val set and it isn't right. In my opinion this is an extreme problem (problem of validation set choosing) to solve in first place in the future!  

## 4. What went into production (best submissions)
My best model performed about 0.55 on the leaderboard. And I've run out of ideas.
I remembered someone at [ODS community](https://ods.ai/) had said about teamplay in competitions that you should look around leaderboard and cooperate with them. I've send requests to two guys above me and [Leheng Li](https://www.kaggle.com/lilelife) replied. His model based on EfficientNet and shows the result of 0.61 on the leaderboard.
I've mixed our models with two approaches: averaging of probabilities and averaging of logits. The first hasn't worked well while the second showed better results than both of our networks.