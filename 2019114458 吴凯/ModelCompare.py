import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    dataset_train = torchvision.datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor())
    dataset_test = torchvision.datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor())
    data_train = dataset_train.data
    X_train = data_train.numpy()
    X_test = dataset_test.data.numpy()
    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))
    Y_train = dataset_train.targets.numpy()
    Y_test = dataset_test.targets.numpy()

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    # 导入数据
    train_x, train_y, test_x, test_y = load_data()
    train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.2, random_state=33)

    # 创建SVM分类器
    model = SVC()
    model.fit(train_x, train_y)
    validation_score = model.score(validation_x, validation_y)
    test_score = model.score(test_x, test_y)
    print('SVM validation accuracy:', validation_score)
    print('SVM test accuracy:', test_score)


    # 创建朴素贝叶斯
    model = MultinomialNB()
    model.fit(train_x, train_y)
    validation_score = model.score(validation_x, validation_y)
    test_score = model.score(test_x, test_y)
    print('朴素贝叶斯 validation accuracy:', validation_score)
    print('朴素贝叶斯 test accuracy:', test_score)

    # 创建决策树
    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    validation_score = model.score(validation_x, validation_y)
    test_score = model.score(test_x, test_y)
    print('决策树 validation accuracy:', validation_score)
    print('决策树 test accuracy:', test_score)

    # KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_x, train_y)
    validation_score = model.score(validation_x, validation_y)
    test_score = model.score(test_x, test_y)
    print('KNN validation accuracy:', validation_score)
    print('KNN test accuracy:', test_score)


