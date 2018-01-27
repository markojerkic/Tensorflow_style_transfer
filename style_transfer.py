import numpy as np
import tensorflow as tf
from scipy.misc import imsave, imread, imresize

import vgg19

# Set a couple of constants
CONTENT_PATH = '/images/content.jpg'
STYLE_PATH = '/images/style.jpg'
CONTENT_LAYER = 'block4_conv1'
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
STYLE_WEIGHT = 1e4
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-4


def load_img(path, shape, content=True):
    img = imread(path)
    if content:
        # If the image is the content image,
        # calculate the shape
        h, w, d = img.shape
        width = int((w * shape / h))
        img = imresize(img, (shape, width, d))
        print('content {}'.format(img.shape))
    else:
        # The style image is set to be the same shape
        # as the content image
        img = imresize(img, (shape[1], shape[2], shape[3]))
        print('style {}'.format(img.shape))
    img = img.astype('float32')
    # Subtract the mean values
    img -= np.array([123.68, 116.779, 103.939], dtype=np.float32)
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def deprocess(img):
    # Remove the fourth dimension
    img = img[0]
    # Add the mean values
    img += np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return img


def calc_content_loss(sess, model, content_img):
    sess.run(tf.global_variables_initializer())
    # Set the input of the graph to the content image
    sess.run(model['input'].assign(content_img))
    # Get the feature maps
    p = sess.run(model[CONTENT_LAYER])
    x = model[CONTENT_LAYER]
    # Euclidean distance
    return tf.reduce_sum(tf.square(x - p)) * 0.5


def gram_matrix(x):
    # Flatten the feature map
    x = tf.reshape(x, (-1, x.shape[3]))
    return tf.matmul(x, x, transpose_a=True)


def calc_style_loss(sess, model, style_img):
    sess.run(tf.global_variables_initializer())
    # Set the input of the graph to the style image
    sess.run(model['input'].assign(style_img))
    loss = 0
    # We need to calculate the loss for each style layer
    for layer_name in STYLE_LAYERS:
        a = sess.run(model[layer_name])
        a = tf.convert_to_tensor(a)
        x = model[layer_name]
        size = a.shape[1].value * a.shape[2].value
        depth = a.shape[3].value
        gram_a = gram_matrix(a)
        gram_x = gram_matrix(x)
        loss += (1. / (4. * ((size ** 2) * (depth ** 2)))) * tf.reduce_sum(tf.square(gram_x - gram_a))
    return loss / len(STYLE_LAYERS)


def main():
    content_img = load_img(CONTENT_PATH, 300)
    style_img = load_img(STYLE_PATH, content_img.shape, content=False)

    vgg = vgg19.VGG()

    with tf.Session() as sess:
        tf_content = tf.constant(content_img, dtype=tf.float32, name='content_img')
        tf_style = tf.constant(style_img, dtype=tf.float32, name='style_img')
        tf_gen_img = tf.random_normal(tf_content.shape)

        # Load the graph
        model = vgg.create_graph(tf_content)

        loss = 0
        loss += CONTENT_WEIGHT * calc_content_loss(sess, model, tf_content)

        loss += STYLE_WEIGHT * calc_style_loss(sess, model, tf_style)

        loss += TV_WEIGHT * tf.image.total_variation(model['input'])

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(tf_gen_img))

        # For this kind of use case, the limited memory BFGS performs the best
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                           options={'maxiter': 500})

        global step
        step = 0

        def update(l):
            # Function to print loss
            global step
            if step % 100 == 0:
                print('Step {}; loss {}'.format(step, l))
            step += 1

        optimizer.minimize(sess, fetches=[loss], loss_callback=update)
        imsave('/output/output.jpg', deprocess(sess.run(model['input'])))


if __name__ == '__main__':
    main()
