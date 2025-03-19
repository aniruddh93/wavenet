# Implementation of Wavenet Model

This repo provides implementation of wavenet model and the parallel wavenet as described in the [WaveNet paper](https://arxiv.org/abs/1609.03499) and [Parallel Wavenet paper](https://arxiv.org/abs/1711.10433). 
For training I used the LJ speech dataset from [Kaggle](https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset)


* wavenet_model.py: Implementation of base wavenet model (as described in wavenet paper)

* train_wavenet.py: Training code for wavenet model. Implements p(x_t/x_1, ..., X_t-1) in two different ways: (a) 256-Categorical with Softmax  and (b) Mixture of logistic distribution

* parallel_wavenet.py: Implementation of Parallel wavenet model. Parallel wavenet uses a student model (Ps) and a trained teacher model (Pt) and minimizes the KL-div b/w them. P(x_t/x_0...x_t-1) is modelled as mixture of logistic distribution.
