import numpy as np
import tensorflow as tf

def read_and_decode(filename): # 读入tfrecords
    filename_queue = tf.train.string_input_producer([filename],shuffle=True)#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64])  #reshape为64*64的1通道图片
    img = tf.cast(img, tf.float32)#在流中抛出img张量
    label = tf.cast(features['label'], tf.float64) #在流中抛出label张量
    #labels = tf.one_hot(label,2)
    return img, label
