## 初始拉曼样本示例

![Image text](https://github.com/852569069/code/blob/master/paper%20data/%E5%A4%8D%E6%9D%82%E6%A0%B7%E6%9C%AC/images/%E5%9B%BE%E7%89%871.png)

## 网络设计思路
    拉曼数据与图像相比，相当于只有一个维度上有数值，因此采用一维卷积。同时采用ResNet网络，每个resnet block内部包含4个卷积层，
    卷积核大小依次为[1,3,3,5],采用不同大小的卷积核能使网络看到不同层次的特征。第一个卷积层通道数为64，在每个block内部的通道
    数为100。由于拉曼数据中特征在于其峰值数据，所以每次卷积都做pading，而且步长都设置为1。最后连接两个全连接层。所有网络包含
    1+4*4+2层，共19层。

## 结果
训练10000次之后，能达到95%正确率

![Image text](https://github.com/852569069/code/blob/master/paper%20data/%E5%A4%8D%E6%9D%82%E6%A0%B7%E6%9C%AC/images/%E5%9B%BE%E7%89%872.png)
## 网络结构

![Image text](https://github.com/852569069/code/blob/master/paper%20data/%E5%A4%8D%E6%9D%82%E6%A0%B7%E6%9C%AC/images/graph_large_attrs_key%3D_too_large_attrs%26limit_attr_size%3D1024%26run%3D%20(1).png)

## 预测示例：
 

