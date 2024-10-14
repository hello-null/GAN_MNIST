# GAN_MNIST
基于生成对抗网络(GAN)生成手写数字

# code文件夹
code文件是Pytorch代码，其中相关的文件路径需要自己手动改变。

# fake_image文件夹
是本人运行代码"生成器"所产生的虚假手写图像。

# loss_curve文件夹
代码运行结束绘制的"生成器"和"鉴别器"损失曲线。

# model_path文件夹
模型权重文件，使用torch.load('./Generator_epoch_80.pth')加载。

# data文件夹
保存的是MNIST数据集，CSV格式   MNIST官网：http://yann.lecun.com/exdb/mnist/
