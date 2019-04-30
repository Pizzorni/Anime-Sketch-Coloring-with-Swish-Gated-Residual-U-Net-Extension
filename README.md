# Anime Sketch Coloring with Swish Gated Residual U Net

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

We set out to produce an open-source implementation of this paper and attempt to reproduce their results. Our implementation is on github and can be found [here](https://github.com/pradeeplam/Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet).

| ![whoops2](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img2.png) | 
|:--:| 
| *Results from original paper [3]* |

<p align="center">
  <img src="https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img2.png">
</p>

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
| *Loss Function* |


Symbol meanings:
* T<sup>u</sup> = *u*th image in output collection
* C = RGB ground truth image
* &phi;<sup>j</sup> <sub>l</sub>
* S<sup>l</sup>
* &lambda;<sub>l</sub> = Weight for *l*th layer loss

This is the summation of the “per-pixel loss” (when l = 0 and &phi;<sub>0</sub>(C) = C) and the “perceptual loss” (when l > 0 and &phi;<sub>1</sub>(C) = the 2nd activation from VGG’s lth convolutional layer). Incorporating the perceptual loss helps capture higher level features and multiplying each layer by the input grayscale image Sl forces the network to focus more on the non-line areas, as in a grayscale image, black (the color of the lines) is represented by 0’s and white by 255.

As the network outputs a “collection” of 9 possible ways to color in the image, the minu operation minimizes loss for the best image in this collection.

## The missing pieces
There were several details not included in the paper that the paper authors clarified for us by email (thanks so much to Xin Chen!). 
The biggest clarification was related to the loss: slowly incorporates losses from all images in the collection, not just the one with the minimum loss, as described above.

When only minimizing the minimum loss image, we got results like this (first image is sketch, second is ground truth RGB image, the rest are generated by the network):

| ![whoops8](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img8.png) | 
|:--:| 
| Only one ouput learns |

As can be seen in this image, only one image was improving at a notable rate. The authors noted they used a similar method as an implementation of [4], where the loss is a weighted sum between the mean collection image loss and the min:
~~~~
# loss_sum is an array of shape [9, 1] containing the per-image loss
loss_min = tf.reduce_sum(tf.reduce_min(loss_sum, reduction_indices=0))
loss_mean = tf.reduce_sum(tf.reduce_mean(loss_sum, reduction_indices=0))
loss = loss_min * 0.999 + loss_mean * 0.001
~~~~
We also encountered a strange situation where many of our images had a sepia-tone, water-color washed out feeling to them:


| ![whoops9](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img9.png) | 
|:--:| 
| Anime meets watercolor |

We adjusted our loss function slightly to take the mean along the image difference channel axis, followed by calculating the mean along the row and column dimensions, rather than simply calculating the sum along the row, column and channel axis. This fixed the issue.

## Training
The images used for training were collected from the “Safebooru” online anime dataset which is a “safe” subset of the “Danbooro” dataset (Although, would still highly not recommend viewing in public). We initially ran these images through a cartoonization filter provided by OpenCV but noticed that the resulting images had less defined edges less than in the original paper.
The authors were able to point us to the method they had used: Github.

The network took roughly 2 days to train on a server with a 3 GHz Xeon Gold 6154 CPU and a Nvidia Titan XP GPU with 12GB of RAM.

The loss progression was as follows:

| ![whoops10](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img10.png) |
|:--:| 
| *Training Loss* |

## Final result
Compare our final results with some of the worse GAN results
Talk about clean-cut-edges and vibrantness

## Extension
Dealing with users is almost always the biggest challenge a project faces. Our target demographic includes mangaka and Manga/Anime enthusiasts, some of the most demanding and particular individuals. We decided it would be wise to allow the user some control over how the images get colored. Our intent was to allow for some interactive coloring after the initial automatic coloring, while still utilizing our network.

We’ve made strides towards accomplishing this. We began by extracting and visualizing the activations for a particular image, and very quickly learned why neural networks are called black boxes. We could initially, sort of, understand the kinds of features that were being extracted. However, as we went deeper in to the network, it very quickly devolved into meaningless pixels.

| ![img11.png](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img11.png) |
|:--:|
| *Convolutional Layer Activations* |

We decided that we didn't need to understand, as long as it worked. So we took an image, fed it through the network, manually modified the output, and back propagated with a higher learning rate. Our goal was to learn which filters at what levels were responsible for different features. Given an image and its output, we manually re-colored the hair of all the outputs to a solid color, fed it back in, and kept track of how many of the filters changed and how they changed. Due to the skip connections inherent in the network and the large number of filters, the answer is a lot of filters changed, and in very different ways numerically. Visually, we couldn't see the difference. It was hard to draw any sort of meaningful conclusion from the raw numerical data, so we instead experimented with changing filters by hand and seeing what happened. This approach taught us a valuable lesson, trying to arbitrarily modify learned parameters in a network leads to horrible outputs with a high confidence rate.
