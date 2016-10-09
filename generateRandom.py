import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

generator = load_model('models/generator_200.h5')
noiseInput = np.random.uniform(-1,1,100)
outputImage = generator.predict(np.array([noiseInput]))

plt.imshow(outputImage[0][0], interpolation = 'nearest', cmap='gray_r')
plt.show()