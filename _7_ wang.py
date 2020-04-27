import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result"""

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
xs = tf.placeholder(tf.float32, [None,784])
ys = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1, 28, 28, 1])#n个28x28通道为1的图片，-1表示n根据参数自己匹配

#conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])#5x5的卷积核，通道数1，这样的卷积核32个
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)#output size 14x14x32

#conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#func1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
#[n_samples,7,7,64] ->> [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#func2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
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

sess= tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/", sess.graph)
#important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    result,_ = sess.run([merged, train_step], feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    writer.add_summary(result,i)
    acc = sess.run(accuracy, feed_dict={xs:mnist.test.images, ys:mnist.test.labels, keep_prob:0.5})
    if i % 50 == 0:
        print(acc)
