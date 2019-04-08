import sys
import numpy as np
import tensorflow as tf
from timeit import timeit


class Embedder:

    def __init__(self, protobuf_file_path):
        # Based on code from:
        # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-
        # and-serve-it-with-a-python-api-d4f3596b3adc
        with tf.gfile.GFile(protobuf_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')

        self.graph = graph

    def embed(self, images):
        image_input = self.graph.get_tensor_by_name('prefix/input:0')
        phase_train = self.graph.get_tensor_by_name('prefix/phase_train:0')
        embedding = self.graph.get_tensor_by_name('prefix/embeddings:0')

        with tf.Session(graph=self.graph) as sess:
            return sess.run(embedding, feed_dict={
                image_input: images,
                phase_train: False,
            })


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Running Embedder on its own requires a protobuf file')
    embedder = Embedder(sys.argv[1])
    print(embedder.embed(np.random.rand(10, 150, 150, 3)))
