# DCGAN

Keras implementation of the following paper on MNIST database.

*Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).*  
[link to paper](https://arxiv.org/pdf/1511.06434v2.pdf)

### Dependencies
* Keras
* Numpy
* matplotlib
* sklearn ( used only for shuffling the data )

### Usage

* `dcgan.py`, main file.

* `generateRandom.py`, uses the saved trained model `generator_200.h5` inside the `models` folder to generate images.

* `metrics` folder contains the discriminator loss and generator loss after every epoch saved in numpy's npy format.

### Results

* Generated images after the final epoch

![](/images/generated_after_200_epoch.png)


* GIF of the network learning the handwritten digits after every 5 epoch

![](/images/dcgan.gif)

### Note
* Using batch normalization as suggested in the paper did not work as expected. Do let me know if I have erred.
* The data is normalized before being fed into the network
* I have concatenated both the train and val data for the train dataset thus 70000 samples of 28*28 each.
* While runnning `generateRandom.py` you might get an error `initNormal not a valid initializations` or something like that. Keras does not save the user initialized functions in the model, to resolve this error, add the following in `python/site-packages/keras/initializations.py`. This ensures that all the weights are initialized from a zero centered normal distribution with standard deviation 0.02.

        def initNormal(shape, name=None):
          return normal(shape, scale=0.02, name=name)
          
