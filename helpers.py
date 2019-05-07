import math
from PIL import Image
import face_recognition
import numpy as np


def crop_to_face(image, image_dimension):
    as_array = np.array(image)
    possible_bounds = face_recognition.api.face_locations(as_array)

    # If multiple faces are found, choose the first arbitrarily
    face_bounds = list(possible_bounds[0])

    # The face_recognition and PIL libraries take input in different formats.
    # Rotate the results for compatibility.
    rotated = face_bounds[-1:] + face_bounds[:-1]
    image = image.crop(rotated)

    embedding_image = resize_image(image, image_dimension)
    return Image.fromarray(embedding_image[0].astype('uint8'), 'RGB')


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

    embedding_image = np.pad(embedding_image,
                             ((0, 0), x_pad, y_pad, (0, 0)),
                             'constant')
    return embedding_image
