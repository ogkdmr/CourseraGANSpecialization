#Downsides of the GANs

* Lack of intrinsic evaluation metrics. 
* Unstable training, there are two moving parts (WGANs, 1-Lipschitz help with this)
* No density estimation of the leant P(X|Y) distribution. That means, you don't know what's an
anomaly within what GAN generated.
* Invertability is not straightforward (you need a 3rd encoder model to map the generated output 
back to its corresponding noise vector.)


#Alternatives to GANs

##Variational Autoencoders (VAE)
They try to minimize the divergence between the generated and real distributions. 
- They produce lower quality results than GANs. 

+ Easier to invert the generated images.
+ Stable training but slower.
+ Has a density estimation of the latent space distribution. 

They try to minimize the divergence between the generated and real distributions. 
- They produce lower quality results than GANs. 

+ Easier to invert the generated images.
+ Stable training but slower.
+ Has a density estimation of the latent space distribution. 
a
