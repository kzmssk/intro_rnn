import tensorflow as tf

class Dense(object):
    def __init__(self, in_dim, out_dim, act_func):
        
        self.W = tf.Variable(tf.truncated_normal((in_dim, out_dim)))
        self.b = tf.Variable(tf.zeros((out_dim)))
        self.f = act_func

    def __call__(self, x):
        return self.f(tf.matmul(x, self.W) + self.b) 
