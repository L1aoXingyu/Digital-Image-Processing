## Digital Image Processing HW2

### Target: Produce a "morph" animation of one input face into another person's face

### 监测人脸关键点并进行三角剖分
通过Dlib库，我们使用python作为binding，然后下载预训练的权重，这样我们导入图片，就能够得到图片中面部监测的关键点，同时使用opencv库中的`cv2.Subdiv2D()`，插入一个一个的关键点，能够得到关键点的三角剖分，这个三角剖分叫 Delaunay 三角，其有一个特点，就是每个三角形的外接圆都不包括其他的特征点。

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-89df77293c5a3a76.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-6de362e4dc427f60.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

当然我们可以对现在的三角剖分进行face morph，但是这样处理的话，背景就不能参与处理了，只有脸能够显现，所以为了将背景也一起处理，我们在上面增加一些关键点，比如图片的四个角和中心，为了体现出脸整体的轮廓，还可以在脸周围加一些特征点，比如脖子，额头，耳朵等等，最后得到的结果如下。

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-5f5e38f6b4c886a7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-7fe79d28ed855cab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 建立三张图片对应的三角形索引
通过两张图片对应特征点的加权求和，我们能够得到合成图片的特征点，这些特征点都是一一对应的，我们需要找到三张图片中一一对应的三角形。方法特别简单，因为所有的点都是对应好的，所以我们只需要在一张图的特征点上找到所有 Delaunay 三角形，并将三角形对于特征点的索引找到，我们就能够找到三张图片对应的三角形。而在一张图片上找到所有的三角形，我们通过`sub_div.getTriangleList()`得到所有可能的三角形三个点的坐标，然后判断他们是否是位于图像内，如果是的话，我们就去遍历所有的点，找到与它距离小于1的点，这样我们就认为这是两个相同的点，然后我们就能够找到所有位于图像内的三角形三个点的索引，这个索引同样适用于另外两张图。

### 构造仿射变换，并构建 face morph
找到了所有图片对应的三角形之后，遍历三角形构建仿射变换，如果输入的图片是img1和img2，我们先通过img1的三角形三个点和face morph图片对应的三角形三个点得到仿射变换矩阵，通过`cv2.getAffineTransform()`能够得到，但是注意这里需要在一个矩形上构建，所以需要先使用`cv2.boundingRect()`构建一个三角形的外接矩形，然后求出三角形三个坐标相对于矩阵的坐标进行仿射变换的求得。

分别得到两张图对于 face morph 图片的仿射变换之后，我们能够通过`cv2.warpAffine()`应用这个仿射变换，而这一步同样需要在一个矩形上，所以我们先通过`cv2.fillConvexPoly()`构建一个mask，然后在img1和img2上取矩形区域进行仿射变换，然后通过mask去掉三角形外面点的变换。

这里进行前五个三角形的构建我们能够得到下面的结果。


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-f0f34a258141d075.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后经过整张图片所有三角形的变换，我们能够得到下面的结果。

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-c499d910555c2b69.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 不同结果的展示
通过设置0.2, 0.4, 0.5, 0.6, 0.8我们能够得到下面不同的结果。

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-a0c8ae8e0d5916a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-8d70c5ce5f1eabfe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-c499d910555c2b69.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-b26b90e3fae2e45d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-18c0768940b459b6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


