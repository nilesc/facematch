import math
import numpy as np

def resize_image(image, image_dimension):
    scale_factor = image_dimension / max(*image.size)
    embedding_image = image.copy().resize((int(image.size[0] * scale_factor),
                          int(image.size[1] * scale_factor)))
    embedding_image = np.array(embedding_image)
    embedding_image = np.expand_dims(embedding_image, 0)
    x_pad = (0, 0)
    if embedding_image.shape[1] < image_dimension:
        difference = image_dimension - embedding_image.shape[1]
        split = math.ceil(difference / 2.0)
        x_pad = (split, split)
    
    y_pad = (0, 0)
    if embedding_image.shape[2] < image_dimension:
        difference = image_dimension - embedding_image.shape[2]
        split = math.ceil(difference / 2.0)
        y_pad = (split, split)
    embedding_image = np.pad(embedding_image, ((0, 0), x_pad, y_pad, (0, 0)), 'constant')
    return embedding_image
