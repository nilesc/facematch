import tensorflow as tf
from keras.models import load_model

model = load_model('pose_weights/biwi_model.h5', custom_objects={
        'GlorotUniform': tf.initializers.glorot_uniform,
    })
print(model)
