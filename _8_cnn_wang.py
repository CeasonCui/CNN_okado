from __future__ import print_function
import os
import tensorflow as tf
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
import ReadOwnData

batch_size = 32
n_batch = int(8960*3*0.8 / batch_size)
channel = 32
epoch = 200
acc_list = []


def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label
    

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #stride[1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

#define palceholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 64, 64])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 64, 64, 1])

#conv1 layer
W_conv1 = weight_variable([3, 3, 1, 32])#5x5的卷积核，通道数1，这样的卷积核32个
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)#output size 14x14x32

#conv2 layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#conv3 layer
W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#conv4 layer
W_conv4 = weight_variable([3, 3, 128, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

#func1 layer
W_fc1 = weight_variable([4*4*256, 2])
b_fc1 = bias_variable([2])
#[n_samples,7,7,64] ->> [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1, 4*4*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#func2 layer
W_fc2 = weight_variable([2, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) #loss
    tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

img, label = ReadOwnData.read_and_decode("triangle_and_others_train.tfrecords")
img_test, label_test = ReadOwnData.read_and_decode("triangle_and_others_test.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)
img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)


sess= tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/", sess.graph)
#important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    val, l = sess.run([img_batch, label_batch])
    l = one_hot(l,2)

    result,_ = sess.run([merged, train_step], feed_dict={xs: val, ys: l, keep_prob: 0.5})
    writer.add_summary(result,i)
    l_test = one_hot(label_test,2)
    acc = sess.run(accuracy, feed_dict={xs:img_test, ys:l_test, keep_prob:0.5})
    if i % 50 == 0:
        print(acc)
