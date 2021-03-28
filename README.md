# Conditional-Generative-Adversarial-Network


## Generative Modeling
*A generative model describes how a dataset is generated, in terms of a probabilistic model. By sampling from this model, we are able to generate new data.*


## Hierarchy Of Generative models
![Hierarchy Of Generative Models](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/Generative%20models%20hierarchy.png)


## Examples of Generative model applications
![Generative modelling use-cases](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/GAN%20images.png)


## GAN Architecture workflow
![Architecture Of GAN](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/GAN%20Architecture.jpeg)


## How a GAN works?
### *The principle - generator v/s discriminator*
The working principle of a GAN is a ***mini-max game*** between two neural networks called the generator and discriminator. The generator tries to fool the discriminator by generating distributions similar to real-domain, say real-looking images, while the discriminator labels the data distribution as real or fake (as you would have seen in a classifier). The generator learns from the feedback it gets from the discriminator and updates itself to produce as real a data distribution as it can.


## Individual Losses

### Generative Loss
The generator tries to **minimize** the loss function given below. Here ***D ( G ( z<sup>(i)</sup> ) )*** is the output of the classifier(discriminator) for the generated image's distribution ***G ( z<sup>(i)</sup> )*** for the i<sup>th</sup> training example. Thus,  ***log ( 1 - D ( G (z<sup>(i)</sup>) ) )*** would correspond to the probability that the discriminator is **not fooled** by the generated image and classified it correctly as a fake image. So, the generator learns by trying to minimize its failure to fool the generator.
![Generator Loss](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/Generator%20Loss.jpeg)

### Discriminative Loss
The discriminator tries to **maximize** the function given below. As we can see that the second term is the same as the function minimized by the generator. Thus, it represents the maximization of the probability of classifying a fake generated image correctly as a fake one. The first term in turn represents the ability of the discriminator to correctly classify a real image. Thus, maximizing this function would train the discriminator as required by us.
Note: In the actual model, the loss function is always minimized, so a negative sign is placed before the loss function term of the discriminator. 
![Discriminator Loss](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/Discriminative%20Loss.jpeg)


The generator G and the discriminator D are jointly trained in a two-player minimax game formulation. The overall minimax objective function, resulting due to combination of the above equations is:
![Objective Function of GAN](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/minmax%20objective%20function.png)


# CGAN - Conditional Generative Adversarial Network
In a normal GAN, only the generation of data distribution from noise occurs, there is no control over modes of the data to be generated. Thus, we can get an object of distribution only and it would not be specific. Only its belongingness to the data distribution would be ensured by our GAN model.

This is tackled by Conditional GANs, which are trained to produce a specific image belonging to the data distribution !!

While only noise is given as input normally to GANs, CGANs along with taking data as input, also take labels as additional input parameters. Now, the noise coupled with the label as a pair is expected to generate an output corresponding to the label, and training of the CGAN occurs in this manner. The discriminator is also fed input data along with labels similar to the generator, to make real-image(data) production easier.

Thus we can generate any specific-featured data from distribution by the use of CGANs, by just including the required labels as input to our model, thus making CGANs a very powerful generative tool.
![CGAN Architecture](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/CGAN%20architecture.png)


## Objective function of a two-player minimax game for conditional GAN
![Objective function of CGAN](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Images/CGAN%20Objective%20Function.png)


## Setup Instructions
You can either download the repo or clone it by running the following in cmd prompt
```
$ https://github.com/Huzaib/Conditional-Generative-Adversarial-Network.git
```
Further, just run the jupyter notebook


## GIF of images generated during training 
![Training GIF](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Training%20GIF/mnist.gif)


## Few Final Generated Results
![Zero](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Evaluation%20Images/0.png)
![Two](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Evaluation%20Images/2.png)
![Three](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Evaluation%20Images/3.png)
![Five](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Evaluation%20Images/5.png)
![Six](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Evaluation%20Images/6.png)
![Nine](https://github.com/Huzaib/Conditional-Generative-Adversarial-Network/blob/main/Evaluation%20Images/9.png)


## Resources

- [Generative Modeling](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/ch01.html)

- [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)

- [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

- [NIPS 2016 Tutorial](https://arxiv.org/abs/1701.00160)

- [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)

- [ProFill: High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling](https://zengxianyu.github.io/iic/)

- [GAN Dissection](https://gandissect.csail.mit.edu/)













