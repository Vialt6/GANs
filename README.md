# GANs
Machine learning project, the architecture for proGAN is the same only changes in log.
If you want to run it and see the result use command: tensorboard --logdir logs 

# About Files
the file where separeted just to keep things ordered, the architecture is the same. 

*dcgan-celeba*: DCGAN trained on CelebA dataset

*proGAN-celebhq*: proGAN trained on celeb-HQ dataset (HQ version of CelebA)

*proGAN-celebhq-pretrained*: proGAN using pre-trained weights found on web.

*proGAN-ffhq*: proGAN trained on FFHQ dataset 

*proGAN-ffhq-pretrained*: proGAN trained on ffhq using weights from proGAN trained on celebHQ

In every folder you can find critic.pth and generator.pth that are the weights trained.  
