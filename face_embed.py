import sys
import tensorflow as tf

class Embedder:

    def __init__(self, protobuf_file_path):
        # Based on code from: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
        with tf.gfile.GFile(protobuf_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')

        self.graph = graph

    def embed(self):
        pass

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Running Embedder on its own requires a protobuf file'
    embedder = Embedder(sys.argv[1])
    embedder.embed()
