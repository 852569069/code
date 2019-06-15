import tensorflow as tf
import os
import numpy as np
import pandas as pd

class Raman(object):
    def __init__(self):
        self.class_name=['Diphenhydramine','Erythromycin','Magnesium Sulphate_2241','Metoprolol',
                         'Valproic Acid','Vanillin_2241','Verapamil HCl','Zinc Oxide_2241']
        pass

    def data_gen(self,index):
        train_data=np.load('../9 data/test.npy')
        label=train_data[:,-1][index]
        data=np.expand_dims(train_data[:,:-1][index],axis=2)
        return data,label

    def res_block(self,x,name,pading=False):
        with tf.variable_scope(name):
            conv1d=tf.layers.conv1d(x,100,1,1,activation=tf.nn.relu,padding='same',name='conv1_1')
            conv1d=tf.layers.conv1d(conv1d,100,3,1,activation=tf.nn.relu,padding='same',name='conv1_2')
            conv1d=tf.layers.conv1d(conv1d,100,3,1,activation=tf.nn.relu,padding='same',name='conv1_3')
            conv1d=tf.layers.conv1d(conv1d,100,5,1,activation=tf.nn.relu,padding='same',name='conv1_4')
            if pading:
                x_channel=x.get_shape().as_list()[-1]
                pading_num=(100-x_channel)//2
                x=tf.pad(x,[[0,0],[0,0],[pading_num,pading_num]])
            out=x+conv1d
        return out
    def resnet(self):
        input=tf.placeholder(tf.float32,[None,881,1])
        labels=tf.placeholder(tf.float32,[None])
        labels=tf.cast(labels,tf.int64)
        with tf.variable_scope('resnet'):
            conv1d=tf.layers.conv1d(input,64,3,padding='same',activation=tf.nn.relu)
            #layer 1
            conv1d=self.res_block(conv1d,'layer1',pading=True)
            #layer2
            conv1d=self.res_block(conv1d,'layer2')
            #layer3
            conv1d=self.res_block(conv1d,'layer3')
            #layer4
            conv1d=self.res_block(conv1d,'layer4')
            flat=tf.layers.flatten(conv1d,name='flatten')
            fc1=tf.layers.dense(flat,1000,activation=tf.nn.sigmoid,name='fc1')
            fc2=tf.layers.dense(fc1,9,activation=tf.nn.softmax,name='fc2')
            predict = tf.argmax(fc2, axis=1)
            acc=tf.metrics.accuracy(labels=labels,predictions=predict)
        return input,labels,predict,acc



class_name=['Diphenhydramine','Erythromycin','Magnesium Sulphate_2241',
            'Magnesium Sulphate_2241','Metoprolol',
            'Valproic Acid','Vanillin_2241',
            'Verapamil HCl','Zinc Oxide_2241']


r=Raman()
input_plac,labels_plac,predicts,acc=r.resnet()
sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
saver=tf.train.Saver()
saver.restore(sess,'../ckp paper/ckp9500')
for i in range(100):
    train_data,train_label=r.data_gen(i)
    train_data=np.expand_dims(train_data,axis=0)
    train_label=np.expand_dims(train_label,axis=0)
    out,accu=sess.run([predicts,acc],{input_plac:train_data,labels_plac:train_label})
    print([class_name[np.int32(i)] for i in train_label],[class_name[np.int32(i)] for i in out])

