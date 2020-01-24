# Unsupervised-Optical-Flow

This is a Pytorch implementation of **Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness** [paper](https://arxiv.org/abs/1608.05842).
The Original paper uses **FlownetS** as a backbone supervised model. In this project the original paper pipeline have been implemented, and other supervised models are to be added; 
**Light Flownet** and **PWC NET**

### Contribution:
Since state of the art of supervised optical flow estimation changes frequently. You can contribute to this project by adding the supervised model implementation to 
*models.py* following the syntaxe of the supervised models already implemented. Namely you need to make sure that your forward pass returns a batch of optical flow 
estimation of size *(N, 2, H, W)*, or a tuple of batches of different spatial resolutions *(N, 2, h1, w1)*, *(N, 2, h2, w2)* ...

### Pretrained model:

https://gofile.io/?c=U1XKvN
