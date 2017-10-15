import numpy as np
import tensorflow as tf
from layers import *

def get_random_x(shape, vmin=0.0, vmax=1.0):
    x_np = np.random.random(shape).astype(np.float32)
    return tf.convert_to_tensor(x_np)

def print_tensor(x):
    print(x.shape)

def run(sess, x, y):
    print_tensor(x)
    print_tensor(sess.run(y))

x_2d = get_random_x((1, 2))

dense_layer = Dense(2, 3, tf.sigmoid)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

run(sess, x_2d, dense_layer(x_2d))


    
