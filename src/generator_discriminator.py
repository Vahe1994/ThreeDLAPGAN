'''
Created on February 4, 2018

@author: vage

'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout
from . tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers

# def simple_generator(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
#                          regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
#                          b_norm_finish=False, verbose=False):
#     '''An Generator ( network), which generate latent space .
#     '''

#     if verbose:
#         print 'Building Generator'

#     n_layers = len(layer_sizes)
#     dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)
#     print(layer_sizes)
#     # if n_layers < 2:
#     #     raise ValueError('For an FC decoder with single a layer use simpler code.')

#     for i in xrange(0, n_layers - 1):

#         name = 'decoder_fc_' + str(i)
#         scope_i = expand_scope_by_name(scope, name)
#         print(i)
#         if i == 0:
#             layer = latent_signal

#         layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

#         if verbose:
#             print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

#         if b_norm:
#             name += '_bnorm'
#             scope_i = expand_scope_by_name(scope, name)
#             layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#             if verbose:
#                 print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

#         if non_linearity is not None:
#             layer = non_linearity(layer)

#         if dropout_prob is not None and dropout_prob[i] > 0:
#             layer = dropout(layer, 1.0 - dropout_prob[i])

#         if verbose:
#             print layer
#             print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

#     # Last decoding layer never has a non-linearity.
#     name = 'decoder_fc_' + str(n_layers - 1)
#     scope_i = expand_scope_by_name(scope, name)
#     layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
#     if verbose:
#         print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

#     if b_norm_finish:
#         name += '_bnorm'
#         scope_i = expand_scope_by_name(scope, name)
#         layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#         if verbose:
#             print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

#     if verbose:
#         print layer
#         print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

#     return layer


# def simple_discriminator(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
#                          regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
#                          b_norm_finish=False, verbose=False):
#     '''A decoding network which maps points from the latent space back onto the data space.
#     '''
#     if verbose:
#         print 'Building Decoder'

#     n_layers = len(layer_sizes)
#     dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)
#     print(n_layers)
#     # if n_layers < 2:
#     #     raise ValueError('For an FC decoder with single a layer use simpler code.')

#     for i in xrange(0, n_layers - 1):
#         name = 'decoder_fc_' + str(i)
#         scope_i = expand_scope_by_name(scope, name)

#         if i == 0:
#             layer = latent_signal

#         layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

#         if verbose:
#             print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

#         if b_norm:
#             name += '_bnorm'
#             scope_i = expand_scope_by_name(scope, name)
#             layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#             if verbose:
#                 print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

#         if non_linearity is not None:
#             layer = non_linearity(layer)

#         if dropout_prob is not None and dropout_prob[i] > 0:
#             layer = dropout(layer, 1.0 - dropout_prob[i])

#         if verbose:
#             print layer
#             print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

#     # Last decoding layer never has a non-linearity.
#     name = 'discriminator_sigmoid_' + str(n_layers - 1)
#     scope_i = expand_scope_by_name(scope, name)
#     layer = tf.sigmoid(layer,name=name)
#     if verbose:
#         print name

#     if b_norm_finish:
#         name += '_bnorm'
#         scope_i = expand_scope_by_name(scope, name)
#         layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#         if verbose:
#             print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

#     if verbose:
#         print layer
#         print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

#     return layer


# def Simple_Gan(n_latent_d):
#     ''' Single class experiments.
#     '''
#     # if n_pc_points != 2048:
#     #     raise ValueError()

#     generator = simple_generator
#     discriminator = simple_discriminator

#     n_input = [n_latent_d, 1]

#     generator_args = {'layer_sizes': [n_latent_d, 128,128],
#                     'b_norm': False,
#                     'b_norm_finish': False,
#                     'verbose': True
#                     }

#     discriminator_args = {'layer_sizes': [256, 512,1],
#                     'b_norm': False,
#                     'b_norm_finish': False,
#                     'verbose': True
#                     }

#     return generator, discriminator, generator_args, discriminator_args

def discriminator_smpl(input,scope=None,reuse=False):

    weight_decay=0.001

    layer_sizes=[256,512,1]
    n_layers =len(layer_sizes)
    for i in xrange(0,len(layer_sizes)-1):
        if(i==0):
            layer = input 
        name = 'discriminator_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = fully_connected(layer, layer_sizes[i], activation='relu', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)    
    
    sigm = tf.nn.sigmoid(layer,name=name)
    return sigm, layer

def generator_smpl(input,n_output,reuse=False):


    reuse=False
    scope='generator'
    layer_sizes=[128,n_output]
    n_layers =len(layer_sizes)
    for i in xrange(0,len(layer_sizes)-1):
        if(i==0):
            layer = input 
        name = 'generator_fc_' + str(i)

        scope_i = scope + name

        layer = fully_connected(layer, layer_sizes[i], activation='relu', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)    
    name = 'generator_fc_' + str(len(layer_sizes))
    scope_i = scope + name  
    layer = fully_connected(layer, layer_sizes[n_layers - 1][0], activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)
    return layer