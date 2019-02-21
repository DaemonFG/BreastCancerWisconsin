# BreastCancerWisconsin
逻辑回归做二分类进行癌症预测(基于细胞的属性特征)

- 数据来源：http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

- 数据描述
1. 699条样本，共11列数据，第一列用语检索的id，后9列分别是与肿瘤相关的医学特征，最后一列表示肿瘤类型的数值。
2. 包含16个缺失值，用”?”标出。

- 思路：
1. 获取数据，指定列名
2. 缺失值处理
3. 数据分割
4. 标准化处理
5. 训练数据
6. 做出预测
