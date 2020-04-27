from __future__ import print_function
import os ,sys
import tensorflow as tf
from PIL import Image  
import matplotlib.pyplot as plt 
import numpy as np
import math
#from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# path = 'C:\\Users\cuizh\Desktop\code\python\gm_2\square'
# dirs = os.listdir( path )

# # 输出所有文件和文件夹
# for file in dirs:
#     print(file)


'''
要改的地方：
writer_train
writer_test
引用函数
验证时的两个地址
'''

batch_size = 32
n_batch = 8960*3*0.8 // batch_size
cwd = 'C:\\Users\cuizh\Desktop\code\python\gm_2'
cwd_square = 'C:\\Users\cuizh\Desktop\code\python\gm_2\square'
cwd_ellipse = 'C:\\Users\cuizh\Desktop\code\python\gm_2\ellipse'
cwd_triangle = 'C:\\Users\cuizh\Desktop\code\python\gm_2\\triangle'
classes={'triangle','others'}  #2 classes
writer_train= tf.python_io.TFRecordWriter("triangle_and_others_train.tfrecords") #要生成的文件
writer_test= tf.python_io.TFRecordWriter("triangle_and_others_test.tfrecords") #要生成的文件



square = []
label_square = []

ellipse = []
label_ellipse = []

triangle = []
label_triangle = []

others = []
label_others = []

#step1:获取Image_to_tfrecords.py文件运行生成后的图片路径
    #获取路径下所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中

def get_square_files(ratio):
    for file in os.listdir(cwd_square): #0 : square
        square.append(cwd_square + '\\' + file)
        label_others.append(0)
    for file in os.listdir(cwd_ellipse): #1: others
        others.append(cwd_ellipse + '\\' + file)
        label_others.append(1)
    for file in os.listdir(cwd_triangle):
        others.append(cwd_triangle + '\\' + file)
        label_others.append(1)

    #打印出提取图片的情况，检测是否正确提取
    #print("There are %d roses\nThere are %d sunflowers\n"%(len(roses),len(sunflowers)),end="")

#step2: 对生成图片路径和标签list做打乱处理把roses和sunflowers合起来组成一个list（img和lab）
    # 合并数据numpy.hstack(tup)
    # tup可以是python中的元组（tuple）、列表（list），或者numpy中数组（array)
    # 函数作用是将tup在水平方向上（按列顺序）合并
    image_list = np.hstack((square,others))
    label_list = np.hstack((label_square,label_others))

    #利用shuffle,转置，随机打乱
    temp = np.array([image_list,label_list])    #转换成2维矩阵
    #print(temp)
    temp = temp.transpose()     #转置
    np.random.shuffle(temp)     #按行随机打乱顺序函数
    #print(temp)
    #从打乱的temp中再取出list（img和lab）
    #image_list = list(temp[:,0])
    #label_list = list(temp[:,1])
    #label_list = [int(i) for i in label_list]
    #return  image_list,label_list
    #print(temp)
    #将所有的img和lab转换成list
    all_image_list = list(temp[:,0])    #取出第0列数据，即图片路径
    all_label_list = list(temp[:,1])    #取出第1列数据，即图片标签
    # for a in range(5):
    #     print(all_image_list[a])
    #     print(all_label_list[a])
    #将所得list分为两部分，一部分用来train，一部分用来测试val
    #ratio是测试集比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))  #测试样本数
    n_train = n_sample - n_val    #训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    #tra_labels = list(map(int, tra_labels.split()))
    #tra_labels.astype(int)
#    tra_labels = [int(float(i)) for i in tra_labels]    #转换成int数据类型
    tra_labels = [int(float(i)) for i in tra_labels]
    
    test_images = all_image_list[n_train:-1]
    test_labels = all_label_list[n_train:-1]
#    val_labels = [int(float(i)) for i in val_labels]    #转换成int数据类型
    #test_labels = list(map(int, test_labels.split()))
    #test_labels.astype(int)
    test_labels = [int(float(i)) for i in test_labels]
    # for a in range(5):
    #     print(test_images[a])
    #     print(test_labels[a])
    return tra_images,tra_labels,test_images,test_labels


def get_ellipse_files(ratio):
    for file in os.listdir(cwd_ellipse): #0 : ellipse
        ellipse.append(cwd_ellipse + '\\' + file)
        label_others.append(0)
    for file in os.listdir(cwd_square): #1: others
        others.append(cwd_square + '\\' + file)
        label_others.append(1)
    for file in os.listdir(cwd_triangle):
        others.append(cwd_triangle + '\\' + file)
        label_others.append(1)
    image_list = np.hstack((ellipse,others))
    label_list = np.hstack((label_ellipse,label_others))

    #利用shuffle,转置，随机打乱
    temp = np.array([image_list,label_list])    #转换成2维矩阵
    #print(temp)
    temp = temp.transpose()     #转置
    np.random.shuffle(temp)     #按行随机打乱顺序函数

    all_image_list = list(temp[:,0])    #取出第0列数据，即图片路径
    all_label_list = list(temp[:,1])    #取出第1列数据，即图片标签

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))  #测试样本数
    n_train = n_sample - n_val    #训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]

    tra_labels = [int(float(i)) for i in tra_labels]
    
    test_images = all_image_list[n_train:-1]
    test_labels = all_label_list[n_train:-1]

    test_labels = [int(float(i)) for i in test_labels]

    return tra_images,tra_labels,test_images,test_labels


def get_triangle_files(ratio):
    for file in os.listdir(cwd_triangle): #0 : triangle
        triangle.append(cwd_triangle + '\\' + file)
        label_others.append(0)
    for file in os.listdir(cwd_square): #1: others
        others.append(cwd_square + '\\' + file)
        label_others.append(1)
    for file in os.listdir(cwd_ellipse):
        others.append(cwd_ellipse + '\\' + file)
        label_others.append(1)
    image_list = np.hstack((triangle,others))
    label_list = np.hstack((label_triangle,label_others))

    #利用shuffle,转置，随机打乱
    temp = np.array([image_list,label_list])    #转换成2维矩阵
    #print(temp)
    temp = temp.transpose()     #转置
    np.random.shuffle(temp)     #按行随机打乱顺序函数

    all_image_list = list(temp[:,0])    #取出第0列数据，即图片路径
    all_label_list = list(temp[:,1])    #取出第1列数据，即图片标签

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))  #测试样本数
    n_train = n_sample - n_val    #训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]

    tra_labels = [int(float(i)) for i in tra_labels]
    
    test_images = all_image_list[n_train:-1]
    test_labels = all_label_list[n_train:-1]

    test_labels = [int(float(i)) for i in test_labels]
    
    return tra_images,tra_labels,test_images,test_labels

train,train_label,test,test_label = get_triangle_files(0.2)


train_count=0
for label, img_path in zip(train_label, train):
    print(label)
    print(img_path)
    train_count+=1
    if train_count==6:
        break


for label, img_path in zip(test_label, test):
    #for img_name in os.listdir(cwd_square): 
        #img_path=cwd_square+img_name #每一个图片的地址
    img=Image.open(img_path)
    #print(img)
    img= img.resize((64,64))
    #print(np.shape(img))
    img_raw=img.tobytes()#将图片转化为二进制格式
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    })) #example对象对label和image数据进行封装
    writer_test.write(example.SerializeToString())  #序列化为字符串

writer_test.close()

for label, img_path in zip(train_label, train):
    #for img_name in os.listdir(cwd_square): 
        #img_path=cwd_square+img_name #每一个图片的地址
    img=Image.open(img_path)
    #print(img)
    img= img.resize((64,64))
    #print(np.shape(img))
    img_raw=img.tobytes()#将图片转化为二进制格式
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    })) #example对象对label和image数据进行封装
    writer_train.write(example.SerializeToString())  #序列化为字符串
 
writer_train.close()

# print('over')

#验证二进制文件正确性
cwd = 'C:\\Users\cuizh\Desktop\code\python\gm_2\check'
filename_queue = tf.train.string_input_producer(["triangle_and_others_train.tfrecords"]) #读入流中
cnt = len(list(tf.python_io.tf_record_iterator("triangle_and_others_train.tfrecords")))
print("train件数：{}".format(cnt))
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [64, 64])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example,'L')#这里Image是之前提到的
        img.save(cwd + '\\' + str(i)+'_''train_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)


filename_queue = tf.train.string_input_producer(["triangle_and_others_test.tfrecords"]) #读入流中
cnt = len(list(tf.python_io.tf_record_iterator("triangle_and_others_test.tfrecords")))
print("train件数：{}".format(cnt))
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [64, 64])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example,'L')#这里Image是之前提到的
        img.save(cwd + '\\' + str(i)+'_''test_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)

print ('over')