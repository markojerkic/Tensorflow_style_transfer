import numpy as np
import tensorflow as tf
from scipy.misc import imsave, imread, imresize

import vgg19

CONTENT_PATH = '/images/content.jpg'
STYLE_PATH = '/images/style.jpg'
CONTENT_LAYER = 'block4_conv2'
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
STYLE_WEIGHT = 1e4
CONTENT_WEIGHT = 5e0
TV_WEIGHT = 1e-4


def load_img(path, shape, content=True):
    img = imread(path)
    if content:
        h, w, d = img.shape
        width = int((w * shape / h))
        img = imresize(img, (shape, width, d))
        print('content {}'.format(img.shape))
    else:
        img = imresize(img, (shape[1], shape[2], shape[3]))
        print('style {}'.format(img.shape))
    img = img.astype('float32')
    img -= np.array([123.68, 116.779, 103.939], dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def deprocess(img):
    img = img[0]
    img += np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return img


def calc_content_loss(sess, graph):
    gen = sess.run(graph[CONTENT_LAYER])
    return tf.reduce_sum(tf.square(graph[CONTENT_LAYER] - gen))


def gram_matrix(x):
    x = tf.reshape(x, (-1, x.shape[3]))
    return tf.matmul(x, x, transpose_a=True)


def calc_style_loss(sess, graph):
    loss = 0
    for layer_name in STYLE_LAYERS:
        gen = sess.run(graph[layer_name])
        size = gen.shape[1] * gen.shape[2]
        depth = gen.shape[3]
        res = (1 / 4 * ((size ** 2) * (depth ** 2))) * tf.reduce_sum(
            gram_matrix(graph[layer_name]) - gram_matrix(gen))
        loss += res
    return loss / len(STYLE_LAYERS)


def main():
    content_img = load_img(CONTENT_PATH, 300)
    style_img = load_img(STYLE_PATH, content_img.shape, content=False)
    gen_img = content_img

    vgg = vgg19.VGG()

    with tf.Session() as sess:
        graph = vgg.create_graph(content_img.shape)

        sess.run(tf.global_variables_initializer())
        sess.run(graph['input'].assign(content_img))
        content_loss = calc_content_loss(sess, graph)

        sess.run(tf.global_variables_initializer())
        sess.run(graph['input'].assign(style_img))
        style_loss = calc_style_loss(sess, graph)

        sess.run(tf.global_variables_initializer())
        sess.run(graph['input'].assign(gen_img))
        total_variance_loss = tf.image.total_variation(graph['input'])

        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * total_variance_loss

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B',
                                                           options={'maxiter': 300})

        sess.run(tf.global_variables_initializer())

        global step
        step = 0

        def update(l, i):
            global step
            if step % 50 == 0:
                print('Step {}; loss {}'.format(step, l))
                imsave('/output/img{}.jpg'.format(step), deprocess(i))
            step += 1

        optimizer.minimize(sess, fetches=[total_loss, graph['input']], loss_callback=update)


if __name__ == '__main__':
    main()
