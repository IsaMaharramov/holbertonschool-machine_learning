* [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

* [Convolutional Neural Networks (CNNs) explained](https://youtu.be/YRhxdVk_sIs?si=jZTePOQirrwjUwCa)

* [The best explanation of Convolutional Neural Networks on the Internet!](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8)

* [Machine Learning is Fun! Part 3: Deep Learning and Convolutional Neural Networks](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721)

* [Convolutional Neural Networks: The Biologically-Inspired Model](https://www.codementor.io/@james_aka_yale/convolutional-neural-networks-the-biologically-inspired-model-iq6s48zms)

* [Back Propagation in Convolutional Neural Networks — Intuition and Code](https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199?gi=a1f9bf0551c3)

* [Backpropagation in a convolutional layer](https://medium.com/data-science/backpropagation-in-a-convolutional-layer-24c8d64d8509)

* [Convolutional Neural Network – Backward Propagation of the Pooling Layers](https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/)

* [Pooling Layer](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/#pooling-layer)

* [deeplearning.ai](https://www.deeplearning.ai/)

# The Big Bang of Modern Deep Learning: A Review of AlexNet

<p align="center">
  <img src="The AlexNet Architecture. ResearchGate.jpg" alt="AlexNet Architecture" width="80%">
</p>

## Introduction
Before 2012, the field of computer vision was largely dominated by manual feature engineering—techniques like SIFT or HOG paired with traditional machine learning classifiers. While artificial neural networks were theoretically understood, they were widely dismissed by the broader computer vision community as being too computationally expensive and far too prone to overfitting on small datasets.

Krizhevsky, Sutskever, and Hinton set out to shatter this plateau. The purpose of their 2012 study, ImageNet Classification with Deep Convolutional Neural Networks, was to prove that a deep convolutional neural network (CNN) could achieve record-breaking accuracy on a highly challenging, massive dataset—the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)—by successfully leveraging high-resolution data and the parallel processing power of modern GPUs.

## Procedures
To tackle the 1.2 million high-resolution images in the ImageNet dataset, the researchers built a massive network dubbed "AlexNet."

The architecture consisted of eight learned layers: five convolutional layers (some followed by max-pooling layers) and three fully connected layers, totaling roughly 60 million parameters and 650,000 neurons. To train a model of this unprecedented size, the study introduced several critical innovations:

* Activation Function: They pioneered the use of non-saturating Rectified Linear Units (ReLUs) instead of standard tanh or sigmoid functions, drastically accelerating training time and helping to solve the vanishing gradient problem.

* Hardware Strategy: To handle the immense computational load, they split the network across two GTX 580 3GB GPUs, with specific layers designed to communicate only across certain channels.

* Overfitting Prevention: Given the 60 million parameters, overfitting was a massive hurdle. They combated this using two key techniques: Data Augmentation (extracting random 224x224 patches from the images and applying horizontal reflections and RGB color alterations) and Dropout (randomly zeroing out 50% of the hidden neurons in the fully connected layers to force the network to learn more robust features).

## Results
The results of this study represented a complete paradigm shift for artificial intelligence. On the ILSVRC-2010 test set, AlexNet achieved top-1 and top-5 error rates of 37.5% and 17.0%, significantly outperforming the previous state-of-the-art methods.

More impressively, in the ILSVRC-2012 competition, a variant of this network achieved a winning top-5 test error rate of 15.3%. To put this into perspective, the second-best entry in that competition—which relied on traditional computer vision methods—scored a much higher 26.2% error rate. AlexNet didn't just win the competition; it completely decimated the previous benchmarks.

## Conclusion
The researchers concluded that a large, deep convolutional neural network is highly capable of achieving record-breaking results on highly challenging datasets using purely supervised learning. They also proved that depth is absolutely critical to this success; they found that removing even a single convolutional layer degraded the network's performance. Ultimately, they demonstrated that GPUs, paired with massive datasets and deep architectures, are the definitive future of visual recognition.

## Personal Notes
Reflecting on this paper as someone who regularly configures custom neural network pipelines in PyTorch and works on real-world machine learning deployments, it’s incredible to read the origin of what we now consider standard practice.

Techniques like ReLU activations and Dropout layers are absolute defaults in almost every model I build today, yet this was the paper that mathematically and practically proved their necessity at scale. It is also a great reminder that algorithmic breakthroughs are deeply tied to hardware constraints—splitting the network across two 3GB GPUs was a brilliant piece of practical engineering that directly paved the way for the massive, unified compute clusters we utilize today.
