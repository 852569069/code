import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
path=u'D:\BaiduYunDownload\paper data'
def file_maker(path):
    """
    :arg:create simulation data of raman and save as .npy
    :param path:the path of raman file
    :return:
    """
    root=os.listdir(path)
    all_path=[os.path.join(path,i) for i in root]
    for j,i in enumerate(all_path):
        array=np.load(os.path.join(i,'array_all.npy'))[0]
        filename = str(i).split('\\')[-1]
        if not os.path.exists('./%s' % filename):
            os.makedirs('./%s' % filename)
        for l in range(100):
            new_array=array+np.random.rand(len(array))*100
            # np.save('./%s'%filename+'/_%d'%l,new_array)
        pos='33%d'%(j+1)
        plt.subplot(np.int32(pos))
        plt.plot(range(len(array)),array,)
        plt.title(str(filename))
        print(filename)
    plt.show()

def creat_data(path):
    all_path=os.listdir(path)
    file_path=[os.path.join(path,i) for i in all_path]
    for i in file_path:#i:C:\Users\85256\OneDrive\学习资料\软件学习\深度学习\code\paper data\raman data\compoent_046
        array_all=[]
        inside_path=[os.path.join(i,l) for l in os.listdir(i)]
        for l in inside_path:
            array=np.load(l)
            array_all.append(array)
        all_data=np.vstack(array_all)
        np.save('%s'%i+'/array_all',array_all)
        print(np.shape(array_all))
        print('finished')

class Raman(object):
    def __init__(self,batch_size):
        self.indicator=0
        self.batch_size=batch_size
        self.class_name=['Diphenhydramine','Erythromycin','Magnesium Sulphate_2241','Metoprolol',
                         'Valproic Acid','Vanillin_2241','Verapamil HCl','Zinc Oxide_2241']

    def load_data(self,path):
        root = os.listdir(path)
        all_root = [os.path.join(path, i) for i in root]
        all_data = []
        for l, j in enumerate(all_root):
            data = np.load(j + '/array_all.npy')
            data_result = np.hstack([data,np.zeros([100,1])+l])
            all_data.append(data_result)
        all_data=np.concatenate(all_data,axis=0)
        sample_num=len(all_data)
        all_data=all_data[np.random.permutation(sample_num)]
        train_data=all_data[0:800]
        np.save('../raman data/train.npy',train_data)
        test_data = all_data[100:]
        np.save('../raman data/test.npy',test_data)
        print(np.shape(all_data))
        print("already saved as 'train.npy' and 'test.npy'")

    def data_gen(self):
        train_data=np.load('../raman data/train.npy')
        label=train_data[:,-1]
        data=np.expand_dims(train_data[:,:-1],axis=2)
        end_indicator=self.indicator+self.batch_size
        data_train=data[self.indicator:end_indicator]
        label_train=label[self.indicator:end_indicator]
        self.indicator=end_indicator
        if end_indicator==len(data):
            index=np.random.permutation(len(data))
            data=data[index]
            label=label[index]
            self.indicator=0
        return data_train,label_train

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
        input=tf.placeholder(tf.float32,[self.batch_size,881,1])
        labels=tf.placeholder(tf.float32,[self.batch_size])
        labels=tf.cast(labels,tf.int64)
        with tf.variable_scope('resnet'):
            # inputs=tf.layers.batch_normalization(input)
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
            # fc1=tf.layers.dropout(fc1)
            fc2=tf.layers.dense(fc1,9,activation=tf.nn.softmax,name='fc2')
            predict = tf.argmax(fc2, axis=1)
            loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=fc2))
            acc=tf.metrics.accuracy(labels=labels,predictions=predict)
            tf.summary.scalar('acc',acc[1])
            tf.summary.scalar('loss',loss)
        return input,labels,fc2,loss,acc
    def train_net(self,loss):
        train_op=tf.train.AdamOptimizer(0.005).minimize(loss)
        return train_op
# file_path=r'C:\Users\85256\OneDrive\学习资料\软件学习\深度学习\code\paper data\raman data'
# file_maker(file_path)
r=Raman(50)
input_plac,labels_plac,result,losses,acc=r.resnet()
train=r.train_net(losses)
sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
merged=tf.summary.merge_all()
saver=tf.train.Saver(max_to_keep=1)
write=tf.summary.FileWriter('graph',sess.graph)
for i in range(10000):
  train_data,train_label=r.data_gen()
  # train_label=np.expand_dims(train_label,axis=1)
  out,_,losse,accu,mer=sess.run([result,train,losses,acc,merged],{input_plac:train_data,labels_plac:train_label})

  # if i%500==0:
      # saver.save(sess,'../ckp paper/ckp%d'%i)
  if i % 50== 0:
      # print(train_label)
      write.add_summary(mer, i)
      print(losse)
      print(accu)

