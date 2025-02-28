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
    #xs = tf.placeholder(tf.float32, [None, 64, 64])/255.   # 64x64
    xs = tf.placeholder(tf.float32, [None, 64, 64])
    ys = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)


with tf.name_scope('image_reshape'):
    x_image = tf.reshape(xs, [-1, 64, 64, 1])
    #print(x_image.shape)  # [n_samples, 64,64,1]

## conv1 layer ##
with tf.name_scope('conv1_layer'):
    W_conv1 = weight_variable([3,3, 1,channel]) # patch 3x3, in size 1, out size 32
    b_conv1 = bias_variable([channel])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 64x64x32
    h_pool1 = max_pool_2x2(h_conv1)                          # output size 32x32x32

## conv2 layer ##
with tf.name_scope('conv2_layer'):
    W_conv2 = weight_variable([3,3, channel, channel*2]) # patch 3x3, in size 32, out size 64
    b_conv2 = bias_variable([channel*2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 32x32x64
    h_pool2 = max_pool_2x2(h_conv2)                          # output size 16x16x64
    
# ## conv3 layer ##
# with tf.name_scope('conv3_layer'):
#     W_conv3 = weight_variable([3,3, channel*2, channel*4]) # patch 3x3, in size 64, out size 128
#     b_conv3 = bias_variable([channel*4])
#     h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # output size 16x16x128
#     h_pool3 = max_pool_2x2(h_conv3)                          # output size 8x8x128
#     #cross_entropy3 = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(h_pool3),
#                                               #reduction_indices=[1])) 
# ## conv4 layer ##
# with tf.name_scope('conv4_layer'):
#     W_conv4 = weight_variable([3,3, channel*4, channel*8]) # patch 3x3, in size 128, out size 256
#     b_conv4 = bias_variable([channel*8])
#     h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4) # output size 8x8x256
#     h_pool4 = max_pool_2x2(h_conv4)                          # output size 4x4x256
#     #cross_entropy4 = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(h_pool4),
#                                               #reduction_indices=[1]))   
## fc1 layer ##
with tf.name_scope('fc1_layer'):
    W_fc1 = weight_variable([16*16*channel*2, 2])
    b_fc1 = bias_variable([2])
    # [n_samples, 4, 4, 256] ->> [n_samples, 4*4*256]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*channel*2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    

## fc2 layer ##
with tf.name_scope('fc2_layer'):
    W_fc2 = weight_variable([2, 2])
    b_fc2 = bias_variable([2])
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



img, label = ReadOwnData.read_and_decode("triangle_and_others_train.tfrecords")
img_test, label_test = ReadOwnData.read_and_decode("triangle_and_others_test.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)
img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

init = tf.initialize_all_variables()
t_vars = tf.trainable_variables()
print(t_vars)


#test_writer = tf.summary.FileWriter("logs/test", sess.graph)
cwd = 'C:\\Users\cuizh\Desktop\code\python\gm_2\check'
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator() 
    threads=tf.train.start_queue_runners(sess=sess,coord=coord) 
    for i in range(50):
        for j in range(n_batch):
            val, l = sess.run([img_batch, label_batch])
            # for h in range(batch_size):
            #     image_check = tf.reshape(val[h], [64, 64,1])
            #     sigle_image = Image.fromarray(val[h], 'L')
            #     sigle_image.save(cwd + '\\' + str(j)+'_' + str(h) + '_train_'+str(l[h])+'.jpg')#存下图片

        l = one_hot(l,2)
        _, acc = sess.run([train_step, accuracy], feed_dict={xs: val, ys: l, keep_prob: 0.5})
        loss = sess.run(cross_entropy, feed_dict = {xs: val, ys: l, keep_prob: 1})
        #loss3 = sess.run(cross_entropy3, feed_dict = {xs: val, ys: l})
        #loss4 = sess.run(cross_entropy4, feed_dict = {xs: val, ys: l})
            #print("batch:[%4d] , accuracy:[%.8f]" % (i, acc) )
        print("Epoch:[%4d] , accuracy:[%.8f], loss:[%.8f]" % (i, acc,loss) )
        acc_list.append(acc)
    val, l = sess.run([img_test, label_test])
    l = one_hot(l,2)
    print(l)
    y, acc = sess.run([prediction,accuracy], feed_dict={xs: val, ys: l, keep_prob: 1})
    print(y)
    print("test accuracy: [%.8f]" % (acc))
    
    print (acc_list)
    plt.plot(acc_list)
    plt.savefig("acc.jpg")

    coord.request_stop()
    coord.join(threads)