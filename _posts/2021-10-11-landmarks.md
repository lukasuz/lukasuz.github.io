---
layout: post
title: Facial landmark projection without training
date: 2021-10-11 11:00:00-0400
description: Projecting facial landmarks into the StyleGAN2 latent space without training
---
<!-- This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine. You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`. If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph. Here is an example:

$$
\sum_{k=1}^\infty |\langle x, e_k \rangle|^2 \leq \|x\|^2
$$

You can also use `\begin{equation}...\end{equation}` instead of `$$` for display mode math.
MathJax will automatically number equations:

\begin{equation}
\label{eq:caushy-shwarz}
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
\end{equation}

and by adding `\label{...}` inside the equation environment, we can now refer to the equation using `\eqref`.

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) that brought a significant improvement to the loading and rendering speed, which is now [on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).
 -->

<!-- <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/9.jpg"> -->
 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <video width="100%" controls autoplay loop>
            <source src="{{ site.baseurl }}/assets/img/projection.m4v" type="video/mp4">
            Your browser does not support the video tag.
        </video> 
    </div>
</div>
<div class="caption">
    Combining <i>look</i> (left) and <i>expression</i> (right) into a single image (middle)
</div>



Many of you probably have heard of the possibility of projecting existing face images into the latent space of a pre-trained Generative Adversarial Network (GAN). This is usually achieved through the minimization of a perceptual loss with respect to a latent code: We attempt to replicate a given image with what the network has learned. Luckily, pre-trained StyleGAN models are readily available for this purpose.

But we can also play around with the loss function, which allows us to modify what we actually want to project into the latent space. Recently I did exactly that. Instead of just minimizing w.r.t. to a perceptual loss, I added a facial landmark loss. Now we can generate a face image that exhibits the look of one image while displaying the facial expression of another image. 

The code is based on the original StyleGAN2-ada repo [0]. For projection of facial landmarks, the l2 norm of the landmark heat maps between projection image and target landmark image is minimized, next to the original LPIPS loss [2]. For heat maps of the landmarks, [1] is used. Thus, there are two target images, one for the *look* and one for the *landmarks*. The objective becomes (noise regularization omitted):

$$
loss = \lambda_{lpips} LPIPS(x_{projection}, x_{target\_look}) + HL(x_{projection}, x_{target\_landmark}),
$$

with ***HL*** being the heat map loss defined as 

$$HL(x_1, x_2) = \sum_i^N \lambda_{landmark} \sqrt{(FAN(x_1) - FAN(x_2))^2},$$

where *N*  is the number of pixels, and ***FAN*** is the landmark heat map extraction model which outputs a three-dimensional matrix, where the depth dimension encodes each single landmark. LPIPS as in [1, 2]. Note that $$\lambda_{landmark}$$ is a **vector** containing the weights for each group of landmarks. Groups are for example: Eye brows, eyes, mouth, etc. Check [1] for more info. By tweaking this vector you can determine what facial features you want to project **more strongly** into the generated images. See below for an example.

It is really not perfect, and still has some bugs, but it is fun to play around with. Check it out [here](https://github.com/lukasuz/stylegan2-landmark-projection). I have also included a Google Colab link in the repo, where you can play around with it yourself :).


Huge shout out to the StyleGAN-team and NVIDIA for their work and pre-trained models. Images are from the FFHQ data set.

### References

[0]: Karras, Tero, et al. "Training generative adversarial networks with limited data." *arXiv preprint arXiv:2006.06676* (2020). Code: https://github.com/NVlabs/stylegan2-ada-pytorch

[1]: Bulat, Adrian, and Georgios  Tzimiropoulos. "How far are we from solving the 2d & 3d face  alignment problem?(and a dataset of 230,000 3d facial landmarks)." *Proceedings of the IEEE International Conference on Computer Vision*. 2017. Code: https://github.com/1adrianb/face-alignment

[2]: Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.