"""
逻辑回归：
线性回归的狮子作为的输入，典型的二分类问题
式子同线性回归，也会产生过拟合

输入值横坐标经过sigmoid函数映射为输出值[0, 1]之间的概率，从而将回归问题转化为预测问题
sigmoid图像与y轴相交于0.5，因此默认阈值为0.5

对数似然损失函数：
相对于均方误差只有一个最小值，对数似然损失拥有多个局部最小值，
目前解决不了，只能改善，尽管没有全局最低但，但是效果还是不错的
1、可以多次随机初始化，多次比较最小值结果
2、也可以在求解过程中，调整学习率

sklearn.linear_model.LogisticRegression(penalty=‘l2’, C = 1.0)
Logistic回归分类器
coef_：回归系数
penalty正则化形式
C正则化力度

逻辑回归根据数据样本大小判断01
哪个类别少，判定概率值是正例

应用：广告点击率预测、电商购物搭配推荐
优点：适合需要得到一个分类概率的场景
缺点：当特征空间很大时，逻辑回归的性能不是很好（看硬件能力）

逻辑回归   解决二分类问题   多种二分类场景  参数：正则化力度                 判别模型：k-近邻，决策树，随机森林，神经网络
VS                                                        根据是否存在先验概率可分为
朴素贝叶斯 解决多分类问题   文本分类        没有参数                       生成模型(有先验概率)：隐马尔科夫

实例：良性恶性乳腺癌肿预测
数据来源：http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

数据描述
（1）699条样本，共11列数据，第一列用语检索的id，后9列分别是与肿瘤
相关的医学特征，最后一列表示肿瘤类型的数值。
（2）包含16个缺失值，用”?”标出。

思路：
1. 获取数据，指定列名
2. 缺失值处理
3. 数据分割
4. 标准化处理
5. 训练数据
6. 做出预测
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def logistic():
    """
    逻辑回归做二分类进行癌症预测(基于细胞的属性特征)
    :return: None
    """
    # 构造列标签名
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']

    # 读取数据
    data = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column)

    # 进行缺失值处理
    data = data.replace(to_replace="?", value=np.nan)
    data = data.dropna()

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)  # 正则化力度C=1.0
    lg.fit(x_train, y_train)
    y_predict = lg.predict(x_test)

    print("回归系数", lg.coef_)
    print("准确率", lg.score(x_test, y_test))
    print("召回率",
          classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))  # 真实为正例的样本中预测结果为正例的比例

    return None


if __name__ == '__main__':
    logistic()
