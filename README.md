# Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension

## Introduction/Background
Imagine this:
Having been driven mad from optimizing hyperparameters, you've decided to quit pursuing machine learning and to devote the rest of your life to drawing Japanese-styled comics.

The term they use in "industry" is manga. And those who draw manga are called "mangaka".
And sorry to say this, but your foreseeable future seems bleak at best.

Chances are you move to Japan and start working as an apprentice for an established mangaka. You spend then next many years of your life in a single room apartment, hunched over a knee-height table, and working tirelessly while earning next to nothing. All in an attempt to break into a highly competitive industry.

However, you now have a trick up your sleeve. You can use deep learning to gain a competitive advantage that allows you to pump out much more content than your peers: automatic colorization.

##Previous Works
We looked for an anime colorization approach to implement that is state-of-the-art which does not yet have a public implementation. Many approaches in this area are based on GANs [1, 2], which typically take a long time to train and exhibit unstable effects once trained. We found “Anime Sketch Coloring with Swish-Gated Residual U-Net” [3] by Gang Liu, Xin Chen, and Yanzhong Hu, published in December 2018, which utilizes a U-Net network structure (SGRU) as well as a perceptual loss to generate several possible colorations for the same black and white image sketch. The paper showed many examples of their algorithm outperforming Paintschainer and Style2paints:
![Comparison image from original paper[3]. Image sources from left to right: Sketch, Paintschainer, Style2Paints, SGRU](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img1.png)

| ![whoops](https://raw.githubusercontent.com/Pizzorni/Anime-Sketch-Coloring-with-Swish-Gated-Residual-U-Net-Extension/master/BlogImg/img1.png) | 
|:--:| 
| *Comparison image from original paper[3]. Image sources from left to right: Sketch, Paintschainer, Style2Paints, SGRU* |
