import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity


def load_image(image_path):
    """
    Process the image provided.
    - Open the image
    - Resize the image
    - Convert the image to RGB mode if it's black and white
    """
    input_image = Image.open(image_path)
    

    # Convert black and white image to RGB mode
    if input_image.mode == 'L':
        input_image = input_image.convert('RGB')

    return input_image

# define the path of the images
parent = r'C:\Users\lenovo\Downloads\Joe\s training image\1-Photoroom.png'
child = r'C:\Users\lenovo\Downloads\Joe\s training image\2-Photoroom.png'

# Load images and convert to RGB
image1 = load_image(parent)
image2 = load_image(child)



# Convert images to numpy arrays
image_array1 = image.img_to_array(image1)
image_array2 = image.img_to_array(image2)

# Stack single channel image three times to create RGB image
rgb_image1 = np.stack((image_array1,) * 3, axis=-1)
rgb_image2 = np.stack((image_array2,) * 3, axis=-1)



vgg16 = VGG16(weights='imagenet', include_top=False, 
              pooling='max', input_shape=(256, 48, 3))

# print the summary of the model's architecture.
vgg16.summary()


for model_layer in vgg16.layers:
  model_layer.trainable = False





# Get image embeddings using VGG16 model
image_embedding1 = vgg16.predict(rgb_image1)
image_embedding2 = vgg16.predict(rgb_image2)

# Calculate similarity score
similarity_score = cosine_similarity(image_embedding1, image_embedding2).reshape(1,)
print(similarity_score)







