import os
import tensorflow as tf
import numpy as np
example_z=[[0]*10]
director='C:\\Users\\ZEUS TASK\\desktop'
os.chdir(director)
import othhelper as helper
from glob import glob
import time
import cv2
import matplotlib.pyplot as plt
model_path = "modela"
from_checkpoint = True
model_store=model_path+"/model.cp"
#Tune The Hyparameters
real_size = (128,128,3)
z_dim = 10
learning_rate_D =  .00005 
learning_rate_G = 2e-4
batch_size = 64
epochs = 2150
alpha = 0.2
beta1 = 0.5
#Load The Data
data_resized_dir = "resized_data"
#Inputs: Generator and Discriminator Placeholders
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")
    learning_rate_G = tf.placeholder(tf.float32, name="learning_rate_G")
    learning_rate_D = tf.placeholder(tf.float32, name="learning_rate_D")
    return inputs_real, inputs_z, learning_rate_G, learning_rate_D
#Discriminator
def Convolution(inputs,filters,k,s,num):
    N=str(num)
    conv = tf.layers.conv2d(inputs = inputs,
                            filters = filters,
                            kernel_size = [k,k],
                            strides = [s,s],
                            padding = "SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            name='conv'+N)
    batch_norm = tf.layers.batch_normalization(conv,
                                               training = True,
                                               epsilon = 1e-5,
                                                 name = 'batch_norm'+N)
    conv_out = tf.nn.leaky_relu(batch_norm, alpha=alpha, name="conv"+N+"_out")
    return conv_out
def discriminator(x, is_reuse=False, alpha = 0.2):
    with tf.variable_scope("discriminator", reuse = is_reuse): 
        conv1_out=Convolution(x,64,5,2,1)
        conv2_out=Convolution(conv1_out,128,5,2,2)
        conv3_out=Convolution(conv2_out,256,5,2,3)
        conv4_out=Convolution(conv3_out,512,5,1,4)
        conv5_out=Convolution(conv4_out,1024,5,2,5)
        flatten = tf.reshape(conv5_out, (-1, 8*8*1024))
        logits = tf.layers.dense(inputs = flatten,
                                units = 1,
                                activation = None)
        out = tf.sigmoid(logits)
    return out, logits 
#Generator
def Deconvolution(inputs,filters,k,s,num,is_train):
    N=str(num)
    trans_conv = tf.layers.conv2d_transpose(inputs = inputs,
                                  filters = filters, kernel_size =
                                  [k,k], strides = [s,s], padding =
                                  "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                name="trans_conv"+N)
    batch_trans_conv = tf.layers.batch_normalization(inputs = trans_conv, training=is_train, epsilon=1e-5, name="batch_trans_conv"+N)
    trans_conv_out = tf.nn.leaky_relu(batch_trans_conv, alpha=alpha, name="trans_conv"+N+"_out")
    return trans_conv_out
def generator(z, output_channel_dim, is_train=True):
    with tf.variable_scope("generator", reuse= not is_train):
        fc1 = tf.layers.dense(z, 8*8*1024)
        fc1 = tf.reshape(fc1, (-1, 8, 8, 1024))
        fc1 = tf.nn.leaky_relu(fc1, alpha=alpha)
        trans_conv1_out = Deconvolution(fc1,512,5,2,1,is_train)
        trans_conv2_out = Deconvolution(trans_conv1_out,256,5,2,2,is_train)
        trans_conv3_out = Deconvolution(trans_conv2_out,128,5,2,3,is_train)
        trans_conv4_out = Deconvolution(trans_conv3_out,64,5,2,4,is_train)
        logits = tf.layers.conv2d_transpose(inputs = trans_conv4_out,filters = 3,kernel_size = [5,5],strides = [1,1],padding = "SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name="logits")
        out = tf.tanh(logits, name="out")
        return out
#Discriminator and generator losses(Sum The losses in one)
def model_loss(input_real, input_z, output_channel_dim, alpha):
    g_model = generator(input_z, output_channel_dim)   
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model,is_reuse=True, alpha=alpha)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_model_fake)))
    return d_loss, g_loss
#Optimization
def model_optimizers(d_loss, g_loss, lr_D, lr_G, beta1):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [op for op in update_ops if op.name.startswith('generator')]
    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=lr_D, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=lr_G, beta1=beta1).minimize(g_loss, var_list=g_vars)
    return d_train_opt, g_train_opt
#Show result
def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode, image_path, save, show,example_z=None):
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    if example_z==None:example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])
    samples = sess.run(generator(input_z, out_channel_dim, False),feed_dict={input_z: example_z})
    images_grid = helper.images_square_grid(samples, image_mode)
    if save == True:
        images_grid.save(image_path, 'JPEG')
    if show == True:
        plt.imshow(images_grid, cmap=cmap)
        plt.show()
    return images_grid
#Training
def train(show,example_z,epoch_count, batch_size, z_dim, learning_rate_D, learning_rate_G, beta1, get_batches, data_shape, data_image_mode, alpha):
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], z_dim)
    d_loss, g_loss = model_loss(input_images, input_z, data_shape[3], alpha)
    d_opt, g_opt = model_optimizers(d_loss, g_loss, lr_D, lr_G, beta1)
    i = 0
    z = -1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        num_epoch = 0
        saver.restore(sess, model_store)
        image_path = ""
        imgh=show_generator_output(sess, 1, input_z, data_shape[3], data_image_mode, image_path, False, show,example_z)
    return imgh
#Load the data and train the network
def USEME(show,example_z):    
    dataset = helper.Dataset(glob(os.path.join(data_resized_dir, '*.jpg')))
    with tf.Graph().as_default():imgh=train(show,example_z,epochs, batch_size, z_dim, learning_rate_D, learning_rate_G, beta1, dataset.get_batches,dataset.shape, dataset.image_mode, alpha)
    return imgh
