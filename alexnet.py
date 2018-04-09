#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, num_classes_brand, num_classes_classes, num_classes_year, keep_prob):
        """Create the graph of the AlexNet model.
        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES_BRAND = num_classes_brand
        self.NUM_CLASSES_CLASSES = num_classes_classes
        self.NUM_CLASSES_YEAR = num_classes_year
        self.KEEP_PROB = keep_prob

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        #brand
        flattened = tf.reshape(pool5, [-1, 6*6*256]) #6*6 5*5
        fc6_brand = fc(flattened, 6*6*256, 4096, name='fc6_brand')
        dropout6_brand = dropout(fc6_brand, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7_brand = fc(dropout6_brand, 4096, 4096, name='fc7_brand')
        dropout7_brand = dropout(fc7_brand, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8_brand = fc(dropout7_brand, 4096, self.NUM_CLASSES_BRAND, relu=False, name='fc8_brand')
        
        
        #classes
        fc6_classes = fc(flattened, 6*6*256, 4096, name='fc6_classes')
        dropout6_classes = dropout(fc6_classes, self.KEEP_PROB)

        fc7_classes = fc(dropout6_classes, 4096, 4096, name='fc7_classes')
        dropout7_classes = dropout(fc7_classes, self.KEEP_PROB)

        self.fc8_classes = fc(dropout7_classes, 4096, self.NUM_CLASSES_CLASSES, relu=False, name='fc8_classes')
        
        
        #year
        fc6_year = fc(flattened, 6*6*256, 4096, name='fc6_year')
        dropout6_year = dropout(fc6_year, self.KEEP_PROB)

        fc7_year = fc(dropout6_year, 4096, 4096, name='fc7_year')
        dropout7_year = dropout(fc7_year, self.KEEP_PROB)

        self.fc8_year = fc(dropout7_year, 4096, self.NUM_CLASSES_YEAR, relu=False, name='fc8_year')
        
        
        


    


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)