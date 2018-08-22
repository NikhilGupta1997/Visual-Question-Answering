# Visual-Question-Answering
Recent developments in Deep Learning has paved the way to accomplish tasks involving multimodal learning. Visual Question Answering [(VQA)](http://www.visualqa.org/) is one such challenge which requires high-level scene interpretation from images combined with language modelling of relevant Q&A. Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. This is a PyTorch implementation of one such end-to-end system to accomplish the task.

### Architecture
The learning architecture behind this demo is based on the model proposed in the [VQA paper](http://arxiv.org/pdf/1505.00468v6.pdf).

![Architecure](http://i.imgur.com/2zJ09mQ.png)

The problem is considered as a classification task here, wherein, 1000 top answers are chosen as classes. Images are transformed by passing it through the [VGG-19 model](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d) that generates a 4096 dimensional vector in the second last layer. The tokens in the question are first embedded into 300 dimensional GloVe vectors and then passed through 2 layer LSTMs. Both multimodal data points are then passed through a dense layer of 1024 units and combined using point-wise multiplication. The new vector serves as input for a fully-connected model having a `tanh` and a final `softmax` layer.

## Authors
* [Nikhil Gupta](https://github.com/NikhilGupta1997)
* [Ayush Bhardwaj](https://github.com/Ayushbh)

Course Project under [**Prof. Parag Singla**](http://www.cse.iitd.ac.in/~parags/)
