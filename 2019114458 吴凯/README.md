# CNN   MNIST 数字识别

## 准备工作

安装对应的库 

```
pip install -r requirements.txt
```

并且下载安装相应显卡版本的cuda和cudnn

## 训练

在当前路径终端输入 （训练过程大约1min左右）

```
python CnnTrain.py
```

## 测试

在当前路径终端输入 （测试保存好的模型，大约5s）

```
python Test.py
```

## 模型比较

在当前路径终端输入 

```
python ModelCompare.py
```

注意：优于SVM以及其它算法开销过大，花费时间3min

## 可视化

在当前路径终端输入

```
tensorboard --logdir runs
```

安装命令提示，在谷歌浏览器中输入观察可视化

```
http://localhost:6006/
```

第一个IMAGES中卷积过程可视化
第二个GRAPHS中是模型可视化

![image-20220616010139465](README.assets/image-20220616010139465.png)
