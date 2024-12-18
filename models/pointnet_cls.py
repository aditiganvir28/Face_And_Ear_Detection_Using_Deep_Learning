import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

# tf.compat.v1.disable_eager_execution()

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

# Point Transformer Layer
def point_transformer_layer(inputs, scope):
    with tf.compat.v1.variable_scope(scope):
        batch_size, num_points, _, channels = inputs.shape

        # Compute queries, keys, and values for each point
        queries = tf_util.conv2d(inputs, channels, [1, 1], scope="queries")      # Shape: (batch_size, num_points, 1, channels)
        keys = tf_util.conv2d(inputs, channels, [1, 1], scope="keys")            # Shape: (batch_size, num_points, 1, channels)
        values = tf_util.conv2d(inputs, channels, [1, 1], scope="values")        # Shape: (batch_size, num_points, 1, channels)

        # Compute positional encoding (relative position between points)
        position_encoding = queries - keys  # Calculate relative positions as offsets
        position_encoding = tf.nn.relu(tf_util.conv2d(position_encoding, channels, [1, 1], scope="positional_encoding"))

        # Calculate attention weights based on position and feature similarity
        attention_logits = queries + position_encoding  # Combine feature and positional information
        attention_logits = tf.nn.softmax(attention_logits, axis=-1)  # Softmax to normalize attention scores

        # Weighted sum of values with attention weights
        attention_output = attention_logits * values
        attention_output = tf.reduce_sum(attention_output, axis=2, keepdims=True)  # Aggregate features based on attention weights

        # Residual connection
        return inputs + attention_output
    
# Main Model
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet with attention, input is BxNx3, output Bx40 """
    batch_size = point_cloud.shape[0]
    num_point = point_cloud.shape[1]
    end_points = {}

    # Input Transformation
    # Input Transformation
    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    # Convolutional Layers
    net = tf_util.conv2d(input_image, 64, [1, 3], padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    
    # Apply Point Transformer Layer after initial feature extraction
    net = point_transformer_layer(net, scope="point_transformer1")

    with tf.compat.v1.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=128)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    # Additional Convolution Layers with new architecture
    net = tf_util.conv2d(net_transformed, 128, [1, 1], padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net_transformed, 256, [1, 1], padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1, 1], padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training, scope='conv6', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
    # net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
    net = tf.reshape(net, [batch_size, -1])

    # Fully Connected Layers
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 170, activation_fn=None, scope='fc3')  # Output adjusted for 40 classes

    return net, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']
    K = transform.shape[1]
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
