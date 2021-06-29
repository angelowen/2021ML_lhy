# Anomaly Detection
## Goal
Whether a machine learning model is able to tell a testing image is of the same class as the training images
## Method
* Autoencoder
epochs: 300
training time : 10hrs
CNN(best)
attention(bad)
data augmentation+ AddGaussianNoise(bad)
optimizer: adabound
drop_out
oncycle_lr
## output
Anomaly Value : If the value obtained is higher, the image is more likely to be anomaly.
Classification : we need a threshold to determine whether it is anomaly
## Reference
https://github.com/leaderj1001/Attention-Augmented-Conv2d
https://www.kaggle.com/c/ml2021spring-hw8/leaderboard#score
https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/hw/HW08/HW08.pdf
https://youtu.be/xkpXP4byXqk

