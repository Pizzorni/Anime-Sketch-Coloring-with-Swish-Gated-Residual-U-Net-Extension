# Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension

## Introduction/Background
Imagine this:
Having been driven mad from optimizing hyperparameters, you've decided to quit pursuing machine learning and to devote the rest of your life to drawing Japanese-styled comics.

The term they use in "industry" is manga. And those who draw manga are called "mangaka".
And sorry to say this, but your foreseeable future seems bleak at best.

Chances are you move to Japan and start working as an apprentice for an established mangaka. You spend then next many years of your life in a single room apartment, hunched over a knee-height table, and working tirelessly while earning next to nothing. All in an attempt to break into a highly competitive industry.

However, you now have a trick up your sleeve. You can use deep learning to gain a competitive advantage that allows you to pump out much more content than your peers: automatic colorization.

## Previous Works
We looked for an anime colorization approach to implement that is state-of-the-art which does not yet have a public implementation. Many approaches in this area are based on GANs [1, 2], which typically take a long time to train and exhibit unstable effects once trained. We found “Anime Sketch Coloring with Swish-Gated Residual U-Net” [3] by Gang Liu, Xin Chen, and Yanzhong Hu, published in December 2018, which utilizes a U-Net network structure (SGRU) as well as a perceptual loss to generate several possible colorations for the same black and white image sketch. The paper showed many examples of their algorithm outperforming Paintschainer and Style2paints:

| ![whoops](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img1.png) | 
|:--:| 
| *Comparison image from original paper[3]. Image sources from left to right: Sketch, Paintschainer, Style2Paints, SGRU* |

We set out to produce an open-source implementation of this paper and attempt to reproduce their results.

| ![whoops2](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img2.png) | 
|:--:| 
| *Results from original paper [3]* |

## Network Architecture
The model is based on an U-net architecture. Typically in image-to-image tasks, it is often necessary to encode the image and then apply some non-linear transformations to the encoding. Then, we decode the this result back to an image. This is done with deconvolutions, but images generated this way are often blurry because of loss of information in the encoding process. U-nets fix this by utilizing skip-layers that directly transfer feature maps from each encoder level to its symmetric decoder level (this also means that the resolution of those feature maps must be the same). An example of a U-net can be seen in the figure below (from [3]).

| ![whoops3](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img3.png) | 
|:--:| 
| *U-net structure* |

This paper introduces a novel structure known as swish-gated residual blocks (SGB). These SGBs contain what are called as swish layers. If X is some input, the the swish layer is defined as S(X) = sigmoid(H(X))*X. The SGBs combine these swish layers alongside the typical residual blocks seen in U-nets. Specifically, our residuals layers are just two leaky ReLU convolutional layers with kernel size of 3 and number of filters equal to the number of channels in the input data. If the SGB is in the left branch of the U-net, then we max-pool (with kernel size of 2 and stride of 2) both the output of the swish-layer and the residuals and concatenate both of them together. If the SGB is in the right branch of the U-net, then we then upsample both the residuals and the output of the swish layer and concatenate them alongside the feature map from the skip connection. A figure explaining the SGB can be seen below.

| ![whoops4](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img4.png) | 
|:--:| 
| *SGB* |

The Swish-Gated Residual U-net (SGRU) makes use of 10 SGBs. Five of these SGBs are strung together to make form the encoder part of the U-net while the other five form the decoder. We also add a 1x1 convolution between each SGB which serves to increase the size of the feature map during the encoding process but decrease the size while decoding. The skip connections of the U-net don’t just copy the feature maps to and from the symmetric encoder and decoder levels, instead, they are passed through a swish layer. While skip connections allow the network to have context when reconstructing the output image, the swish layers allow the network to filter information as its being passed through. Finally, the output of the last SGB goes through a couple convolutional layers before finally outputing a collection of images. The output of our model is a tensor with 27 channels. Our model is actually generating 9 RGB images, each with a different style. Our model outputs multiple different potential colored images as this is a one-to-many problem. The diagram below shows the full structure of the SGRU model. Each blue box represents the 4-dimensional tensor after each operation (colored arrows). The number above each blue box represents the number of channels in that tensor. The input to the model is a line-art image which only needs only has one channel. 

| ![whoops5](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img5.png) | 
|:--:| 
| *SGRU Architecture* |

We also used Tensorboard to debug our model. 

| ![whoops6](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img6.png) | 
|:--:| 
| *Tensorflow Graph* |

## Loss Function
After passing the image through the U-Net, a loss is computed based on the “per-pixel loss” and “perceptual loss” of the image. The formalized loss function as seen in the paper is as follows:

| ![whoops7](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img7.png) | 
|:--:| 
|  |

Symbol meanings:
* T<sup>u</sup> = *u*th image in output collection
* C = RGB ground truth image
* &phi;<sup>j</sup> <sub>l</sub>
* S<sup>l</sup>
* &lambda;<sub>l</sub> = Weight for *l*th layer loss
