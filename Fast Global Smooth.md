### Fast Global Image Smoothing Based on Weighted Least Squares

在本篇论文中，主要将全局光滑问题转换为一个加权二次能量函数，最小化这个能量函数从而实现图片光滑的效果，而这个最优化问题，可以转化为一个线性方程组的求解问题，在求解效率和求解时间上都有明显的优势，同时这还是一个全局最优的线性系统。

本次作业主要对论文的基本算法**Separable global smoother for 2D iamge smoothing**进行了复现，使用python语言，为了加快计算速度，使用mxnet.ndarray在GPU上进行运算。





### 1D Fast Global Smoother

对于二维的问题，我们可以将其分解为1维的情况，所以先求解1维的线性系统。

对于1D的情况，可以定义一个WLS能量函数如下
$$
J(u) = \sum_{x} ((u_x^h - f_x^h)^2 + \lambda_t \sum_{i \in N_h(x)} w_{x, i}(g^h)(u_x^h - u_i^h)^2)
$$
需要极小化这个能量函数，相当于$\Delta J(u) = 0$，可以等价于求解下面的线性系统:
$$
(I_h + \lambda_t A_h) u_h = f_h
$$
**使用下面的迭代方法**

首先计算向前方向
$$
\tilde{c_x} = c_x / (b_x - \tilde{c_{x-1}} a_x) \\
\tilde{f^h_x} = (f_x^h - \tilde{f_{x-1}^h} a_x)  /  (b_x - \tilde{c_{x-1}} a_x)  \\
(x = 1, \cdots, W-1)
$$

然后计算向后方向
$$
u_x^h = \tilde{f^h_x} - \tilde{c_x} u_{x+1}^h \quad(x = W - 2, \cdots, 0) \\
u_{W-1}^h = \tilde{f_{W-1}^h}
$$

首先计算相似度权重函数$w_{p, q}(g)$如下


```python
# 定义计算相似度权重函数
def cw_1d(p, q, g, sigma):
    '''
    计算1d上曲线g上不同位置p和q的相似性，使用sigma作为一个范围度量
    g: ndarray，W
    p: int
    q: int
    sigma: float
    '''
    norm = nd.norm(g[p] - g[q])
    return nd.exp(-norm/sigma)
```

为了避免streaking，每次迭代需要修改$\lambda$的值，计算公式为
$$
\lambda_t = \frac{2}{3} \frac{4^{T - t}}{4^T -1 }\lambda
$$
计算lambda的代码如下


```python
# 每次迭代计算lambda
def compute_lamb(t, T, lamb_base):
    return 1.5 * 4**(T-t) / (4 ** T - 1) * lamb_base
```

#### 求解一阶全局光滑线性系统

求解1D情况主要使用上图的(3)和(4)这两个数学公式，代码实现如下


```python
# 求解一维的全局光滑线性系统
def compute_1d_fast_global_smoother(lamb, f, g, sigma, ctx):
    '''
    求解1d fast global smoother，给出lambda, f和g，进行前向和反向的递归求解
    '''
    w = f.shape[0]
    _c = nd.zeros(shape=w-1, ctx=ctx)
    _c[0] = -lamb * cw_1d(0, 1, g, sigma) / (1 + lamb * cw_1d(0, 1, g, sigma))
    _f = nd.zeros(shape=f.shape, ctx=ctx)
    _f[0] = f[0] / (1 + lamb * cw_1d(0, 1, g, sigma))
    # 递归前向计算
    for i in range(1, w-1):
        _c[i] = -lamb * cw_1d(i, i + 1, g, sigma) / (
            1 + lamb * (cw_1d(i, i - 1, g, sigma) + cw_1d(i, i + 1, g, sigma)) +
            lamb * _c[i - 1] * cw_1d(i, i - 1, g, sigma))
        _f[i] = (f[i] + _f[i - 1] * lamb * cw_1d(i, i - 1, g, sigma)) / (
            1 + lamb * (cw_1d(i, i - 1, g, sigma) + cw_1d(i, i + 1, g, sigma)) +
            lamb * _c[i - 1] * cw_1d(i, i - 1, g, sigma))
    _f[w-1] = (f[w-1] + _f[w-2] * lamb * cw_1d(w-1, w-2, g, sigma)) / (
            1 + lamb * (cw_1d(w-1, w-2, g, sigma)) +
            lamb * _c[w-2] * cw_1d(w-1, w-2, g, sigma))
    u = nd.zeros(shape=f.shape, ctx=ctx)
    u[w - 1] = _f[w - 1]
    # 递归向后计算
    for i in range(w - 2, -1, -1):
        u[i] = _f[i] - _c[i] * u[i + 1]
    return u.asnumpy()
```

#### 完成了一维的全局光滑线性系统的求解，验证一下其是否有效

首先构造一个一维的波动输入，画出图像如下

```python
x1 = np.random.normal(scale=0.2, size=(100))
x2 = np.random.normal(4, 0.2, size=(100))
x = np.concatenate((x1, x2))
plt.plot(np.arange(x.shape[0]), x)
```

![output_8_1.png](http://upload-images.jianshu.io/upload_images/3623720-2458ab21c9f6cabc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


通过一维全局光滑线性系统的求解，给出参数$\lambda=900, \sigma=0.07$，得到光滑之后的图像如下


```python
u = compute_1d_fast_global_smoother(900, nd.array(x, ctx=mx.gpu(0)), nd.array(x, mx.gpu(0)), 0.07, mx.gpu(0))
plt.plot(np.arange(u.shape[0]), u)
```

![output_10_1.png](http://upload-images.jianshu.io/upload_images/3623720-0ca1b8ea03868df5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


可以明显看出能够对输入的一阶波动进行有效地光滑化，对比于论文里面的其他结果，我们认为fast global smooth具有计算时间短，优化更好的特点。





### 2D image separable fast global smooth

对于一张2D的图片，问题非常简单，分别在水平和竖直上应用1D的solver就能够求解，只是需要注意，在计算$w_{p, q}(g)$的时候，如果是一张RGB的图片，需要在 channel 维度的向量上计算相似度。对于这种可分的方法，虽然很直观，但是会出现streaking artifact的问题，为了解决这个问题，在每一次的迭代中，可以修改$\lambda$的值，因为这个值在每次迭代中对稀疏光滑具有显著的减少。

下面是可分离全局光滑的代码实现，只需要不断在行和列上进行计算即可，同时需要注意，如果是RGB三通道的图片，需要在每个通道上分别计算


```python
#可分离的全局图片光滑
def Separable_global_smoother(f, T, lamb_base, sigma, ctx):
    '''
    全局图片光滑算法，输入是f和g，f是2D image, 3 channels
    '''
    print('origin lamb is {}'.format(lamb_base))
    print('sigma is {}'.format(sigma))
    H, W, C = f.shape
    u = f.copy()
    for t in range(1, T+1):
        # 计算每一步迭代的lambda_t
        lamb_t = compute_lamb(t, T, lamb_base)
        # horizontal
        for y in range(0, W):
            g = u[:, y, :]
            for c in range(C):
                f_h = u[:, y, c]
                u[:, y, c] = compute_1d_fast_global_smoother(lamb_t, f_h, g, sigma, ctx)
        # vertical
        for x in range(0, H):
            g = u[x, :, :]
            for c in range(C):
                f_v = u[x, :, c]
                u[x, :, c] = compute_1d_fast_global_smoother(lamb_t, f_v, g, sigma, ctx)
    return u
```



完成代码之后，我们读入一张图片，进行multi-scale manipulate测试

首先读入原图，大小是200x200


```python
origin_img = Image.open('./jay.jpg')
```


```python
plt.figure(figsize=(5, 5))
plt.title('origin image')
plt.imshow(origin_img)
```

![output_15_1.png](http://upload-images.jianshu.io/upload_images/3623720-050f0470685961c9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



#### 对小尺度的图片做fast global smooth

接着将图片缩小尺度为100 x 100的图片，下面是缩小之后的图片，可以发现图片变得更加模糊，有很多噪点


```python
img_100 = origin_img.resize((100, 100))
plt.figure(figsize=(5, 5))
plt.title('50 x 50 image')
plt.imshow(img_100)
```

![output_17_1.png](http://upload-images.jianshu.io/upload_images/3623720-60248f75e06e2782.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


接着我们进行一次迭代

```python
lamb_base = 30**2
sigma = 255 * 0.03
ctx = mx.gpu(0)
img = np.array(img_100)
img = nd.array(img, ctx=ctx)
```

下面是一次迭代的结果，可以发现图片已经变得比较光滑，但是会出现条纹，具体可以看看下面红框的部分


```python
T = 1
u1 = Separable_global_smoother(img, T, lamb_base, sigma, ctx)
show_u = u1.astype('uint8').asnumpy()
plt.figure(figsize=(5, 5))
plt.title('1 iteration smooth result')
plt.imshow(show_u)
```

![output_20_2.png](http://upload-images.jianshu.io/upload_images/3623720-bdf0e6bce797e0b3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



经过第二次迭代，下面是迭代结果，关注于两张图片的红框部分，可以发现两次迭代具有比较好的改善效果


```python
T = 2
u2 = Separable_global_smoother(img, T, lamb_base, sigma, ctx)
show_u = u2.astype('uint8').asnumpy()
plt.figure(figsize=(5, 5))
plt.title('2 iteration smooth result')
plt.imshow(show_u)
```

![output_22_2.png](http://upload-images.jianshu.io/upload_images/3623720-299e9b81ad2574c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




三面是三次迭代之后的结果，可以发现图片更加光滑了


```python
T = 3
u3 = Separable_global_smoother(img, T, lamb_base, sigma, ctx)
show_u = u3.astype('uint8').asnumpy()
plt.figure(figsize=(5, 5))
plt.title('3 iteration smooth result')
plt.imshow(show_u)
```

![output_24_2.png](http://upload-images.jianshu.io/upload_images/3623720-f1a835effa1c17b0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 对原始图片做 fast global smooth


```python
lamb_base = 30**2
sigma = 255 * 0.03
ctx = mx.gpu(0)
img = np.array(origin_img)
img = nd.array(img, ctx=ctx)
```

经过一次迭代之后的结果，虽然仍然存在条纹，但是比小尺度的图片效果好，这是因为小尺度的图片有更多的噪声和不光滑性


```python
T = 1
u1 = Separable_global_smoother(img, T, lamb_base, sigma, ctx)
show_u = u1.astype('uint8').asnumpy()
plt.figure(figsize=(5, 5))
plt.title('1 iteration smooth result')
plt.imshow(show_u)
```

![output_28_2.png](http://upload-images.jianshu.io/upload_images/3623720-17aff4574c7fe69d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




经过第二次迭代之后的结果


```python
T = 2
u2 = Separable_global_smoother(img, T, lamb_base, sigma, ctx)
show_u = u2.astype('uint8').asnumpy()
plt.figure(figsize=(5, 5))
plt.title('2 iteration smooth result')
plt.imshow(show_u)
```

![output_30_2.png](http://upload-images.jianshu.io/upload_images/3623720-dd286fab83d5f246.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




下面是三次迭代之后的结果，可以发现图片越来越光滑


```python
T = 3
u3 = Separable_global_smoother(img, T, lamb_base, sigma, ctx)
show_u = u3.astype('uint8').asnumpy()
plt.figure(figsize=(5, 5))
plt.title('3 iteration smooth result')
plt.imshow(show_u)
```

![output_32_2.png](http://upload-images.jianshu.io/upload_images/3623720-fb51bbead4de932a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




### 与Guided Image Filter的对比

fast global image smooth是隐式求解，其将问题转移为一个优化问题，然后再转移为一个线性系统的求解，而线性系统的求解有比较成熟的解法，从而在计算效率和时间上有所提升，同时这还是一个全局最优的解法。

反观guided image filter，这篇文章我并没有非常仔细地阅读，通过大概的阅读，我发现其是一种显示的求解，通过局部的线性关系，将guided image的光滑性质转移到源图片上，通过最小化源图片与光滑图片之间的差异，将问题转移为一个岭回归问题(linear ridge regression)，显示计算出双边滤波的数值，通过其进行光滑化。因为其模型的构建是在局部上进行，所以其存在解的局限性。



### 总结

通过本次作业，了解到光滑化图片的一些方法总结，复现了第一篇论文的算法，比较简单，没有过多的阅读论文的扩展内容。对于guided image filter，没有太过仔细的阅读，只是大体看了一下论文的核心想法，所以理解可能还不到位，有一些误解。