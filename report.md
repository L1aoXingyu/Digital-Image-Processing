# DenseNet 论文报告
$Gao Huang^*, Zhuang liu^*,\ Laurens Van der Maaten$

![](https://ws1.sinaimg.cn/large/006tNc79ly1fmbo8ykh0pj30eh0a6q37.jpg)

## 卷积网络的发展
卷积网络大概从 2012 年左右开始迅速发展，不同的网络结构层出不穷，主要的变化是下面的四个部分。

![](https://ws1.sinaimg.cn/large/006tKfTcly1fmcz1yh4fbj30pk0asq6o.jpg)

### 深度加深
目前卷积网络从最简单的 LeNet， AlexNet， VGG 到 ResNet，网络从比较浅层的结构到了比较深层的结构，比如 VGG 只有 16 层，现在的 ResNet 可以轻松训练到 100 多层，网络层数的增加极大地增强了模型的复杂程度和表现力。

### Filter 大小的改变
最开始的 VGG 的 filter 都是用比较大的 filter，比如 7x7，5x5 的 filter，到后面的 InceptionNet，ResNet 等都是使用 3x3 和 1x1 的 filter，使用比较小的 filter 能极大地减小了使用的参数量。 

### 全连接层的减少
因为全连接层的参数会占据网络中的大部分，所以不断减少网络中的全连接层，目前基本上只有网络的最后一层使用全连接进行分类，而是用全卷积的网络不仅能够节约参数，同时还能应用于多尺度的图片上。

### 层与层之间的连接
以前的网络都是标准的连接，而现在的网络有了更多跨层之间的连接，比如 ResNet，使用跨层连接能够非常好地解决梯度消失和优化的问题。

## 深度学习面临的挑战
目前深度学习面临着一下三个主要的挑战。

### 优化困难
虽然现在网络有很多改进，但是还是存在着优化困难的问题，特别是比较复杂的网络优化难度更高，存在着梯度消失和梯度爆炸的问题。

### 泛化性能
越是复杂的网络越是有着过拟合的风险，所以需要思考更多的办法来使得网络有更好的泛化能力。

### 资源消耗
目前卷积网络都非常深，所以训练和预测都需要非常长的时间，同时也需要非常多的 GPU 资源，所以训练的代价非常大。

## 随机深度网络
随机深度网络希望能够解决一下的几个问题
- 加快训练速度
- 网络易于优化
- 能够有效抑制过拟合

### 随机深度网路的结果
![](https://ws4.sinaimg.cn/large/006tKfTcly1fmd1k43k4lj30nt06gdhv.jpg)

这是随机深度网路用于 ResNet 的基本结构，在每一层都有概率被扔掉，这样在训练的时候，每个 mini batch 的网络层数都不一样，所以这被称为随机深度网络。

![](https://ws4.sinaimg.cn/large/006tKfTcly1fmd1lrw3s4j30oz0cndkp.jpg)

在训练的时候，每次网络的深度都不一样，但网络都比较浅，优化起来非常容易。在预测的时候使用所有的层，保证性能良好。

随机深度网络有下面三个优点：
- 训练非常快，因为每次都会扔掉一些层，所以网络会变得更浅
- 可以有更深的结构，因为网络会扔掉一些层，所以网络能够比原本的结构更深
- 更好的泛化性能，因为每次训练的时候都可以看做一个不同的弱分类器，那么最后集成在一起做预测，能够有更好的性能

![](https://ws3.sinaimg.cn/large/006tKfTcly1fmd1sd2gizj30pa0baaeh.jpg)

上面是随机深度网络的层连接之间的图示，如果每个层都有概率 p 被保留，那么我们可以计算出所有的层之间连接的概率，可以发现每一层都有概率被连起来，但是越远的层被连起来的概率越小。

## DenseNet
DenseNet 的提出就是借鉴于上面的思想，网路有一个更紧致的结构，同时有着更好的泛化性能。

![](https://ws1.sinaimg.cn/large/006tKfTcly1fmd22vdu2xj30ni0b6q6s.jpg)

上面是 DenseNet 的网络，每个层与层之间都有一个连接，但是连接是通过 channel 做 拼接，而不是像 ResNet 做求和。

### Growth Rate
为了网络的 channel 不会太大，我们希望卷积之后的 channel 都比较小，且固定成一个常数，这个常数被称为增长率。

![](https://ws1.sinaimg.cn/large/006tKfTcly1fmd28hbuydj30iu0bk77m.jpg)

下面是一次前向传播的示意图

![](https://ws3.sinaimg.cn/large/006tKfTcly1fmd29al06kj30o705kq4s.jpg)

其中一个单层的结构如下

![](https://ws1.sinaimg.cn/large/006tKfTcly1fmd29vesstj30gi094jt9.jpg)

非常简单，输入的图片经过一个 Batch Norm，在经过非线性激活函数 ReLU，最后经过一个 3x3 的卷积层输出 k channels 的特征。

### Bottleneck Layer
网络在不断加深的过程中就算使用比较小的 Growth Rate，特征也越来越大，所以有一个瓶颈层，将 channel 的维度降低。

![](https://ws2.sinaimg.cn/large/006tKfTcly1fmd2cvagkuj30nu08qwhd.jpg)

瓶颈层也非常简单，主要是使用 1x1 的卷积进行降维，将原先的 lxk 的维度降到 4xk，然后再通过一个单层的结构降维到 k。这个是单层的替代，主要是为了信息的损失比较小。

### Architecture

![](https://ws3.sinaimg.cn/large/006tKfTcly1fmd2hldsdoj30ox07qgoq.jpg)

最后放上 DenseNet 的整体结构，里面一共有 3 个 Dense Block，每个 Block 内部进行 dense 连接，中间使用一个过渡层进行 size 的减小。

### Advantage
DenseNet 有着下面的几个优势
- 更强的梯度回传，因为最后一层和前面的层存在直接的连接关系，所以可以直接将误差会传到前面的层，梯度能够很大程度上保留。
- 参数的使用更少，一般卷积网络需要的参数量是 $O(C^2)$，其中 C 表示channel 的大小。而 DenseNet 的参数量只有 $O(l \times k^2)$ 的量级，其中 $k \ll  C$，所以非常有效地节省了参数。
- 泛化性能更强，因为网络最后分类的特征不仅仅有高级的特征，还有比较底层的特征，这保证了特征的多样性，从而使得分类具有更好的泛化能力。

## Result
![](https://ws3.sinaimg.cn/large/006tKfTcly1fmd2rb0mx9j30ou0c2tdq.jpg)

从结果可以发现 DenseNet 比 ResNet 有更好的泛化性能和准确率，同时参数的使用更少。

## Code
官方的[实现](https://github.com/liuzhuang13/DenseNet)

我自己的[实现](https://github.com/SherlockLiao/cifar10-gluon)
