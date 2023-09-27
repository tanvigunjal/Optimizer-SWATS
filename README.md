# Optimizer-SWATS

SWATS-Optimizer: Switching from Adam to SGD. [[1]](#1)

# Overview
Adaptive optimization methods like RMSprop, Adagrad or Adam [[2]](#2) generalizes poorly despite their superior training performance when compared to Stochastic gradient descent (SGD)[[3]](#3). Adaptive optimization methods perform well in the initial training stage but lack in performance in later stages due to unstable and non-uniform learning rate at the end of training. Hence, SGD generalizes better when compared to adaptive methods.

# Experiment
To perform analysis we used ResNet-34 and DenseNet architectures on <a href = "https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10 </a> dataset. Dataset has 60000 images, which is divided into train set (40000 images), validation set (10000 images) and test set(10000 images). We then compared the results for three optimizers: Adam, SGD and SWATS, with learning rate 0.001 and threshold 1e-5.

The SWATS technique, proposed by Keskar and Socher, involves transitioning from the Adam optimization algorithm to Stochastic Gradient Descent (SGD) under the condition that the discrepancy between the bias-corrected projected learning rate and the projected learning rate falls below a specified threshold ϵ.The determination of the projected learning rate involves projecting the update of Stochastic Gradient Descent (SGD) onto the update of Adam. The switch is applied universally, meaning that if any layer within the network transitions to Stochastic Gradient Descent (SGD), all layers will undergo the transition to SGD.

# Requirements
 Critcal Requirements:
 <li> Python >= 3.8
 <li> NVidia GPU - V100 or better
 <li >Ubuntu >= 18.04 with CuDNN >= 11.7

 Primary Requirements:
 <li> pytorch 
 <li> torchvision
 <li> numpy
 <li> tqdm
 <li> wandb
 <li> math
 <li> matplotlib
 <li> pandas

# Setup
<li>batch_size = 128</li>
<li>epochs = 150</li>
<li>initial learning rate = 0.01</li>
<li>threshold to switch or eposilon = 10^-5 (In the paper \epsilon = 10^-9)</li>
<li>loss_function = Cross Entropy loss</li>

# References 

<a id="1">[1]</a> 
Improving Generalization Performance by Switching from Adam to SGD.<a href="
https://doi.org/10.48550/arXiv.1712.07628
">Link</a>

<a id="2">[2]</a> 
Adam: A Method for Stochastic Optimization.<a href="
https://doi.org/10.48550/arXiv.1412.6980
">Link</a>

<a id="3">[3]</a> 
An overview of gradient descent optimization
algorithms.<a href="
https://arxiv.org/pdf/1609.04747.pdf
">Link</a>
