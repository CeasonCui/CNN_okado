from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
import os
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
import ReadOwnData
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

acc_list = []
channel = 32
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    with tf.name_scope('weight'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape):
    with tf.name_scope('bias'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    with tf.name_scope('conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    with tf.name_scope('man_pool_2x2'):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('image_reshape'):
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 64,64,1]

## conv1 layer ##
with tf.name_scope('conv1_layer'):
    W_conv1 = weight_variable([3,3, 1,channel]) # patch 3x3, in size 1, out size 32
    b_conv1 = bias_variable([channel])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1)                          # output size 14x14x32

## conv2 layer ##
with tf.name_scope('conv2_layer'):
    W_conv2 = weight_variable([3,3, 32, channel*2]) # patch 3x3, in size 32, out size 64
    b_conv2 = bias_variable([channel*2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2)                          # output size 7x7x64

## fc1 layer ##
with tf.name_scope('fc1_layer'):
    W_fc1 = weight_variable([7*7*64, 10])
    b_fc1 = bias_variable([10])
    # [n_samples, 4, 4, 256] ->> [n_samples, 4*4*256]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
with tf.name_scope('fc2_layer'):
    W_fc2 = weight_variable([10, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



# the error between prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
    tf.summary.scalar('loss',cross_entropy)

with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter("logs/train", sess.graph)
# test_writer = tf.summary.FileWriter("logs/test", sess.graph)
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        #test_result = sess.run(merged, feed_dict={xs: mnist.test.images[:1000], ys: mnist.test.labels[:1000], keep_prob: 1})
        #test_writer.add_summary(test_result, i)
        #train_result = sess.run(merged, feed_dict={xs: mnist.train.images[:6000], ys: mnist.train.labels[:6000], keep_prob: 1})
        #train_writer.add_summary(train_result, i)
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
        acc_list.append(acc)

for i in range(50):
    for j in range(60):
        batch_xs, batch_ys = mnist.train.next_batch(100)
            # for h in range(batch_size):
            #     #check_image = tf.reshape(val[h], [64, 64])
            #     sigle_image = Image.fromarray(val[h], 'L')
            #     #print(check_image.shape)
            #     sigle_image.save(cwd + '\\' + str(i) + '_' + str(j)+'_' + str(h) + '_train_'+str(l[h])+'.jpg')#存下图片

        _, acc = sess.run([train_step, accuracy], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        loss = sess.run(cross_entropy, feed_dict = {xs: batch_xs, ys: batch_ys, keep_prob: 1})
        #print("batch:[%4d] , accuracy:[%.8f], loss:[%.8f]" % (j, acc,loss) )
   # print("Epoch:[%4d] , accuracy:[%.8f], loss:[%.8f]" % (i, acc,loss) )
    acc_list.append(acc)

    acc1 = sess.run([accuracy], feed_dict={xs: mnist.test.images[:1000], ys: mnist.test.labels[:1000], keep_prob: 1})
    print("test accuracy: [%.8f]" % (acc))
    
    print (acc_list)
    plt.plot(acc_list)
    plt.savefig("acc.jpg")