import tensorflow as tf
import tensorflow.contrib.slim as slim
#from tf.contrib.layers import layer_norm



def network_single_branch(input_tensor, input_label, num_feautres=32, 
                    is_training=True, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        dense1 = tf.layers.dense(input_tensor, num_feautres*2*8*8)
        dense2 = tf.layers.dense(input_label, num_feautres*2*8*8)

        relu1, relu2 = tf.nn.relu(dense1), tf.nn.relu(dense2)
        reshape1 = tf.reshape(relu1, [-1, 8, 8, num_feautres*2])
        reshape2 = tf.reshape(relu2, [-1, 8, 8, num_feautres*2])
    
        concat = tf.concat([reshape1, reshape2], axis=-1)

        deconv1 = slim.conv2d_transpose(concat, num_feautres*2, [5, 5], stride = [2, 2])
        batchnorm1 = slim.batch_norm(deconv1, is_training=is_training)
        # output size: 16*16

        deconv2 = slim.conv2d_transpose(batchnorm1, num_feautres, [5, 5], stride = [2, 2])
        batchnorm2 = slim.batch_norm(deconv2, is_training=is_training)
        # output size: 32*32

        output = slim.conv2d_transpose(batchnorm2, 1, [3, 3], activation_fn = None)

        return output
    

'''
First use fully connected networks for classification,
then reshape the ouput of fully connected network and
use convolution layers to enhance image quality.
'''

def network_two_branchs(input_tensor, num_feautres=32, img_size=48, is_training=True):
   
    dense1 = tf.layers.dense(input_tensor, 160)
    relu1 = tf.nn.relu(dense1)
    norm1 = tf.layers.batch_normalization(relu1, training=is_training)
    
    dense2 = tf.layers.dense(norm1, 160)
    relu2 = tf.nn.relu(dense2)
    norm2 = tf.layers.batch_normalization(relu2, training=is_training)
    
    dense3 = tf.layers.dense(norm2, 160)
    relu3 = tf.nn.relu(dense3)
    norm3 = tf.layers.batch_normalization(relu3, training=is_training)
    
    dense4 = tf.layers.dense(norm3, 2)
    prob_out = tf.nn.softmax(dense4) 
        
    avg_pool = tf.reduce_mean(dense4, axis=-1)
    #print(avg_pool.get_shape())
    reshape = tf.reshape(avg_pool, [-1, img_size, img_size, 1])

    conv1 = slim.conv2d_transpose(reshape, num_feautres, [3, 3])
    conv_norm1 = tf.layers.batch_normalization(conv1, training=is_training)
    
    conv2 = slim.conv2d_transpose(conv_norm1, num_feautres, [3, 3])
    conv_norm2 = tf.layers.batch_normalization(conv2, training=is_training)

    img_out = slim.conv2d_transpose(conv_norm2, 1, [3, 3], activation_fn = None)*255

    return prob_out, img_out
    



