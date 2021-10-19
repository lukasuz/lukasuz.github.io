---
layout: post
title: Face animations without training
date: 2021-10-17 12:00:00-0000
description: Extending facial landmark projection to produce animations
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
 

<!-- <div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <video width="100%" controls autoplay loop>
            <source src="{{ site.baseurl }}/assets/img/projection.m4v" type="video/mp4">
            Your browser does not support the video tag.
        </video> 
    </div>
</div>
<div class="caption">
    Combining <i>look</i> (left) and <i>expression</i> (right) into a single image (middle)
</div> -->
*If you do not have any previous knowledge in projecting images into the latent space of StyleGAN, you should read [3] and [5] first.*

In my <a href="{{ site.baseurl }}/blog/2021/landmarks/">previous post</a>, I showed how to project facial landmarks into the latent space of a pre-trained StyleGAN2 while keeping the look of another one (to some extent), i.e. the identity. In this post, I will explain how to expand this approach for generating short face animations in two ways. The first approach keeps a face identity, while the second one does **not** keep a face identity. Allowing for smooth transitions between faces while projecting coherent facial animation into the latent space. Note that no model was trained for this approach and it is purely done through minimization, thus the quality of generation is limited. I will go into the limitations of this methods later in this post.

You can generate your own animations in this <a href="https://colab.research.google.com/drive/1ghSB78uobrRxWRtDSCsXCKEUOkUlpOyH?usp=sharing">notebook</a>. The code itself can be found in this <a href="https://github.com/lukasuz/stylegan2-landmark-projection">repository</a>.

<h3>Preserving the face identity (more or less)</h3>
For this, we will need two inputs. Firstly, a face image, $$x_{look}$$, whose look shall be replicated. Secondly, a sequence of $$i$$ images, $$x_{landmark}^i$$, whose facial landmarks are used for projection. Examples are given below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/look_img.png">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <!-- <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/smile.mp4"> -->
         <video class="img-fluid rounded z-depth-1"  width="100%" controls autoplay loop>
            <source src="{{ site.baseurl }}/assets/img/scream_input.m4v" type="video/mp4">
            Your browser does not support the video tag.
        </video> 
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <!-- <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/smile.mp4"> -->
         <video class="img-fluid rounded z-depth-1"  width="100%" controls autoplay loop>
            <source src="{{ site.baseurl }}/assets/img/scream_output.m4v" type="video/mp4">
            Your browser does not support the video tag.
        </video> 
    </div>
</div>
<div class="caption">
    Example 1: Left the look image (<a href="https://github.com/NVlabs/ffhq-dataset">FFHQ</a>), middle the landmark video (<a href="https://www.pexels.com/video/a-woman-screaming-8724361/">pexels</a>), right the output.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/look_img.png">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <!-- <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/smile.mp4"> -->
         <video class="img-fluid rounded z-depth-1"  width="100%" controls autoplay loop>
            <source src="{{ site.baseurl }}/assets/img/smile_input.m4v" type="video/mp4">
            Your browser does not support the video tag.
        </video> 
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <!-- <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/smile.mp4"> -->
         <video class="img-fluid rounded z-depth-1"  width="100%" controls autoplay loop>
            <source src="{{ site.baseurl }}/assets/img/smile_output.m4v" type="video/mp4">
            Your browser does not support the video tag.
        </video> 
    </div>
</div>
<div class="caption">
    Example 2: Left the look image (<a href="https://github.com/NVlabs/ffhq-dataset">FFHQ</a>), middle the landmark video (<a href="https://www.pexels.com/video/">pexels</a>), right the output.
</div>

For each video frame index $$i$$ we want to find the latent code $$w^i$$, with $$f(w^i) = x^i$$, where $$x^i$$ is a face image and $$f$$ the StyleGAN face generation network, which minimizes a loss function made up of three terms:

$$
    \arg \min_{w^i} \lambda_{1} \mathcal{L}_{lpips}(x_{look}, x^i) + \lambda_{2} \mathcal{L}_{fan}(x_{landmark}^i, x^i, \lambda_{landmark}) + \lambda_{3} \\ \mathcal{L}_{smooth}(w^i, w^{i-1}).
$$

Practically, this is achieved through gradient descent, keeping the network weights fixed, but treating the input $$w^i$$ variable. The first term measures the perceptual similarity between generated images and the target look, see [2]. The second term captures the divergence between facial landmarks of the current generated image and the current target video frame. The last term calculates the similarity between the current latent code and the previous latent code. More specifically, the second term is defined as:

$$
\mathcal{L}_{fan}(x^1, x^2, \lambda_{landmark}) = \sum_i^N \lambda_{landmark} \sqrt{(FAN(x^1) - FAN(x^2))^2},
$$

where *N*  is the number of pixels, and ***FAN*** is the landmark heat map extraction model [1] which outputs a three-dimensional matrix, $$\mathcal{R}^{64x64xc}$$, where the last dimension corresponds to the landmarks and the first two correspond to the spatial dimensions. Note that $$\lambda_{landmark}$$ is a **vector** containing the weights for each group of landmarks. Groups are, for example, Eyebrows, eyes, mouth, etc. Check [1] for more info. By tweaking this vector you can determine what facial features you want to project **more strongly** into the generated images. For example, drop any influence of the eyes, but keep mouth movement.

The smoothness term is defined as the l2 norm between consecutive latent codes:

$$
\mathcal{L}_{smooth}(w^1, w^2) =  ||w^1-w^2||_2.
$$

It helps to keep features stable that are not directly constrained by the perceptual loss or the heat map loss more stable over time. It encourages $$w^i$$ to stay close to its predecessor.

During the first iteration, $$w^i$$ is initialized with $$w^i=\hat{w}$$, which is the mean latent vector from randomly sampled latent codes [3]. It corresponds to the average face based on the data set. Furthermore, we iterate for the first image $$iter_{first}=300$$ times. Even though $$1000$$ iterations are suggested in [3], it yields decent results here and is preferably due to my limited resources.

For consecutive frame indices, $$i > 0$$, we initialize the current latent code with the previous latent code, $$w_i = w_{i-1}$$. As the next frame should be relatively similar to the previous one, we can reduce the number of iterations for consecutive frames, $$iter_{consecutive}=50$$. Note that I choose these values empirically, based on a trade-off between quality and generation time. With more iterations (i.e more computing power), the results probably would get better.

For further smoothing the video, subframes are generated which effectively double the number of frames per second. This is possible due to the interpolation capabilities of the latent space. A subframe is generated by linear interpolating neighbouring latent codes $$w^{i,i+1} = \frac{1}{2}w^i + \frac{1}{2}w^{i+1}$$, the corresponding frame is generated by $$f(w^{i,i+1}) = x^{i,i+1}$$. Theoretically, we could increase the sampling rate between latent codes for more smoothness, but a single interpolation point seems to do the job quite well.

<h4>Limitations</h4>

Several problems exist with this approach:
- First of all, as you can see the cropped face video is noisy and therefore quite wiggly, which results in this wiggling being present in the projected video as well. Temporal smoothing will most likely help here.
- Second, the identity is not kept in the course of the sequence. I believe that the extracted landmarks actually carry some signal from the image, thus we project conflicting information into the latent space. An "optimal" representation of one image's look while expressing the facial landmarks from another one does not seem to be completely possible with this approach. This was already visible from the previous post. Perhaps, the introduction of a Person Reid loss would help. Furthermore, shape and pose are entangled in the facial landmarks which results in changing the shape of eyes and mouth in the projected image, when comparing with the input look image.
- Thirdly, facial landmarks are ambiguous. For example, a laughing face's landmarks might be identical to a face with bigger lips. 
- Next, the weighting factors of each term have a big influence on the results and usually have to be adapted for each video somewhat. While a stronger smoothness factor might keep less constrained features (e.g. lighting and hair) constant over the sequence, it actually can result in the landmarks not being projected at all.
- Lastly, it is relatively slow, especially because I am running it on Colab. A sequence of a second takes around 20 minutes to generate.

This approach can not compete quality-wise with s.o.t.a. StyleGAN-based face animation approaches. Nevertheless, I am happy with the quality of the results. As no training of networks is necessary, it works on Colab, only a single look image is necessary, and no dataset-based analysis of the latent space is necessary.

Due to not being able to preserve identity, I prefer the second option explained in the following section.

<h3>Disregarding the face identity</h3>

<div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
        </div>
    <div class="col-sm mt-3 mt-md-0">
         <video class="img-fluid rounded z-depth-1"  width="100%" controls autoplay loop>
            <source src="{{ site.baseurl }}/assets/img/animation_wo_identity.m4v" type="video/mp4">
            Your browser does not support the video tag.
        </video> 
    </div>
    <div class="col-sm mt-3 mt-md-0">
    </div>
    </div>
<div class="caption">
    Video projection without a look image. I used a sequence of myself here.
</div>

In this case, we do a simple modification to the minimization term, where we drop the perceptual similarity term, but add a face fidelity term:

$$
    \arg \min_{w^i} \lambda_{2} \mathcal{L}_{fan}(x_{landmark}^i, x^i, \lambda_{landmark}) + \lambda_{3} \mathcal{L}_{smooth}(w^i, w^{i-1}) + \lambda_{4} \mathcal{L}_{fidelity}(w^i, \hat{w}).
$$

The face fidelity term is defined as the l2 norm between the current latent code and the mean latent code, similar to the smoothness term. This term is necessary, as the quality deteriorates very quickly when we do not constrain the latent space. This is based on the truncation idea proposed in [4], where truncation of $$w$$ results in images with better quality while losing some variation in faces. 

Looking at the video, we can see that the frames are a little bit smoother when compared to the previous case. The face also smoothly changes the identity in the course of the sequence. However, the faces exhibit relatively similar features, i.e. male, beard, curly hair. This is interesting to see, as I also have a beard and I used a sequence of myself in this use case. This further strengthens the assumption that the output of the FAN network carries some signal from the input image.

I am looking forward to trying this out in the future with the improved StyleGAN3 model.

Again, a huge shout out to the StyleGAN-team and NVIDIA for their work and pre-trained models.

### References

[0]: Karras, Tero, et al. "Training generative adversarial networks with limited data." *arXiv preprint arXiv:2006.06676* (2020). Code: https://github.com/NVlabs/stylegan2-ada-pytorch

[1]: Bulat, Adrian, and Georgios  Tzimiropoulos. "How far are we from solving the 2d & 3d face  alignment problem?(and a dataset of 230,000 3d facial landmarks)." *Proceedings of the IEEE International Conference on Computer Vision*. 2017. Code: https://github.com/1adrianb/face-alignment

[2]: Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018. 

[3]: Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[4]: Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

[5]: Abdal, Rameen, Yipeng Qin, and Peter Wonka. "Image2stylegan: How to embed images into the stylegan latent space?." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

