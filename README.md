<h3 align="center">
    <p>Code Implement of <a href="https://github.com/iamownt/LSTM-DeepFM">
    A Data-driven Self-supervised LSTM-DeepFM Model for Industrial Soft Sensor</p>
</h3>

##  Quick Tour
```

```
##  Contents

### Data Preprocess
[Download sample dataset](https://www.kaggle.com/podsyp/production-quality) and put them in the dataset folder.

````
# Some guides of binning are in datapreprocess/Kmeans_Analysis.ipynb
# Some guides of permutation feature importance are in datapreprocess/PIMP_Tutorial.ipynb
python datapreprocess/DataGenerator.py # generate file for training and evaluation
python datapreprocess/train_pimp.py # get the null importance distribution and actual importance distribution
python datapreprocess/visualize_pimp.py # visualize the importance distribution of PIMP
````
###Pretraining 
````
python pretrain/unsupervised_pretraining.py # for unsupervised pretraining
python pretrain/selfsupervised_pretraining.py # for self-supervised pretraining
````
###Finetuning
````
python finetuning/supervised_finetuning.py # for supervised finetuning
````

##  Questions
**How to compare performance with your model**

***We will add a module to quickly use the proposed methods, as shown in Quick Tour.***

**The model does not perform well on some datasets**

***Some parameters of the model, such as the size of the hidden layer, have a great impact on the performance. 
For example, in Debutanizer Dataset, a smaller hidden layer help the model generalize well.***

**The related papers of self-supervised learning**

***Self-supervised learning has achieved great success in natural language processing and computer vision, such as [Bert](https://arxiv.org/abs/1810.04805), [MAE](https://arxiv.org/abs/2111.06377)
. Especially there is a amount of unlabeled data, the task of self-supervised learning is more conducive to mining hidden relationships in the Industrial Big Data.***

**Is the FM module useful or will it impair performance**

***Our initial goal is to achieve fusion learning of various industrial data characteristics, 
so the ability of FM to extract discrete features is important. However, if there are few discrete features 
or the feature importance is low, the performance of the model may be reduced. FM module can play 
an integrated role. If the fusion learning performance decreases, SciPy can be used to find the optimal fusion weight. 
Therefore, the performance of LSTM-DeepFM will be better than that of single LSTM-Deep.***




