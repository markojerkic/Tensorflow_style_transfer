import h5py
import tensorflow as tf

URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1' \
      '/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 '


def maybe_download(path, url):
    p = tf.keras.utils.get_file(path, url, file_hash='253f8cb515780f3b799900260a226db6')
    return p


class VGG:
    def __init__(self):
        # Download weights
        weights_path = maybe_download('weights.h5', URL)
        self.weights_file = h5py.File(weights_path, 'r')
        # Print to start a new line in output terminal
        print()
        # Create a list of the names of all layers in the VGG network
        self.layer_names = [name for name in self.weights_file]

    def load_weights_and_biases(self, name):
        dataset = self.weights_file[name]
        weights = dataset[name + '_W_1:0'][:, :, :, :]
        biases = dataset[name + '_b_1:0'][:]
        return weights, biases

    def conv_layer(self, prev_layer, curr_layer_name):
        w, b = self.load_weights_and_biases(curr_layer_name)
        logits = tf.nn.conv2d(prev_layer, w, [1, 1, 1, 1], 'SAME') + b
        return tf.nn.relu(logits)

    def pool_layer(self, prev_layer):
        logits = tf.nn.avg_pool(prev_layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        return logits

    def create_graph(self, img):
        graph = {'input': tf.Variable(img, dtype=tf.float32)}

        last_layer_name = 'input'
        for name in self.layer_names:
            kind = name[7:11]
            if kind == 'conv':
                graph[name] = self.conv_layer(graph[last_layer_name], name)
                last_layer_name = name
            elif kind == 'pool':
                graph[name] = self.pool_layer(graph[last_layer_name])
                last_layer_name = name
        return graph
