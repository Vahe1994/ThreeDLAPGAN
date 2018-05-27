'''
Created on August 28, 2017

@author: optas
'''

import os.path as osp
import tensorflow as tf

MODEL_SAVER_ID = 'models.ckpt'


class Neural_Net(object):
#addding self.sess (Vage)
    def __init__(self, name, graph, sess=None):
        if graph is None:
            graph = tf.get_default_graph()

        self.graph = graph
        self.name = name
        self.sess = sess
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        

        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

    def is_training(self):
        is_training_op = self.graph.get_collection('is_training')
        return self.sess.run(is_training_op)[0]
