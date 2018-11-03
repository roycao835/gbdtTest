import numpy as np
from scipy.special import expit
import pandas as pd
from numpy import *

def init(y):
    return np.log(float(np.sum(y)) / (y.shape[0] - np.sum(y)))

def negative_gradient(y, pred):
    if type(pred) in [int, float]:
        return y - expit(pred)
    else:
        return mat(y) - expit(mat(pred.T))

def sigmoid(x):
    return 1.0 / ( 1 + np.exp(-x))

def concat_mat(x, y):
    x_list = x.tolist()
    y_list = y.tolist()
    result = []
    for i in range(len(x_list)):
        temp = x_list[i] + [y_list[0][i]]
        result.append(temp)

    return mat(result)


class DumbGbdt():
    def init_prior(y):
        return np.log(float(np.sum(y)) / (y.shape[0] - np.sum(y)))

    def __init__(self, tree_num=2, learning_rate=0.1):
        self.tree_num = tree_num
        self.leanrning_rate = learning_rate
        self.trees = []

    def fit(self, x, y):
        y_pred = init(y)

        for i in xrange(0, self.tree_num):
            tree = DumbTree()
            residual = negative_gradient(y, y_pred)
            # tree.fit(x, residual)
            tree.fit(concat_mat(mat(x), mat(residual)))
            self.trees.append(tree)
            y_pred += self.leanrning_rate * tree.predict(mat(x))

    def predict(self, x):
        result = 0.
        for tree in self.trees:
            result += self.leanrning_rate * tree.predict(x)
        return sigmoid(result)


class DumpTreeNode():
    def __init__(self, split_feature=0, split_value=0, left=None, right=None):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right

    def print_tree(self):
        stack = []
        stack.append(self)
        i = 0
        while len(stack) > 0:
            print("level -------" + str(i))
            curr_level = []
            while len(stack) > 0:
                curr_level.append(stack.pop())
            for node in curr_level:
                if node is None:
                    continue
                print(node)
                stack.append(node.left)
                stack.append(node.right)
            i += 1

    def __str__(self):
        return "feature: %s, value: %s" % (self.split_feature, self.split_value)

def loadDataSet(fileName):
    """loadDataSet(解析每一行，并转化为float类型)
        Desc：该函数读取一个以 tab 键为分隔符的文件，然后将每行的内容保存成一组浮点数
    Args:
        fileName 文件名
    Returns:
        dataMat 每一行的数据集array类型
    Raises:
    """
    # 假定最后一列是结果值
    # assume last column is target value
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将所有的元素转化为float类型
        # map all elements to float()
        # map() 函数具体的含义，可见 https://my.oschina.net/zyzzy/blog/115096
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


class DumbTree():
    def __init__(self, max_depth=3, min_error_diff=0, min_leaf_size=5):
        self.max_depth = max_depth
        self.min_error_diff = min_error_diff
        self.min_leaf_size = min_leaf_size
        self.tree = None

    def variance(self, data):
        return var(data[:, -1]) * data.shape[0]

    def find_split(self, data, depth):
        if (depth <= 0):
            return None, mean(data[:, -1])
        if len(set(data[:, -1].T.tolist()[0])) == 1:
            return None, mean(data[:, -1])
        m, n = data.shape

        error = self.variance(data)

        best_error, best_feature, best_value = inf, 0, 0

        for feature in range(n - 1):
            print(feature)
            for potential_value in set(data[:, feature].T.tolist()[0]):
                left, right = self.split(data, feature, potential_value)
                if (left.shape[0] < self.min_leaf_size) or (right.shape[0] < self.min_leaf_size):
                    continue
                new_error = self.variance(left) + self.variance(right)
                # print ("new_error: " + str(new_error))
                if new_error < best_error:
                    best_feature = feature
                    best_value = potential_value
                    best_error = new_error
        if (error - best_error) < self.min_error_diff:
            print (str(error) + ", " + str(best_error))
            return None, mean(data[:, -1])

        return best_feature, best_value

    def split(self, data, feature, value):
        return data[nonzero(data[:, feature] <= value)[0], :], data[nonzero(data[:, feature] > value)[0], :]

    def create_tree(self, data, depth):
        root = DumpTreeNode()
        split_feature, split_value = self.find_split(data, depth)
        if split_feature is None:
            print("cant split anymore")
            root.split_feature = None
            root.split_value = split_value
            return root
        root.split_feature = split_feature
        root.split_value = split_value
        left, right = self.split(data, split_feature, split_value)
        root.left = self.create_tree(left, depth - 1)
        root.right = self.create_tree(right, depth - 1)
        return root

    # return root node of the tree
    def fit(self, data):
        self.tree = self.create_tree(mat(data), self.max_depth)
        return self.tree

    def predict_result(self, node, oneData):
        if node.split_feature is None:
            return node.split_value
        if oneData[0, node.split_feature] <= node.split_value:
            if node.left.split_feature is None:
                return node.left.split_value
            else:
                return self.predict_result(node.left, oneData)
        else:
            if node.right.split_feature is None:
                return node.right.split_value
            else:
                return self.predict_result(node.right, oneData)

    def predict(self, data):
        data = mat(data)
        m = len(data)
        result = mat(zeros((m, 1)))

        for i in range(m):
            result[i, 0] = self.predict_result(self.tree, mat(data[i]))

        return result


def loss(pred, y):
    result = 0
    pred_list = pred.tolist()
    y_list = y.tolist()

    for i in range(len(y_list)):
        result += (y_list[i][0] - pred_list[i][0]) * (y_list[i][0] - pred_list[i][0])
    return result


def test():
    # test regression tree
    data = loadDataSet("testTree2.txt")
    tree = DumbTree()
    tree.fit(data)
    tree.tree.print_tree()
    print("loss: " + str(loss(tree.predict(data), mat(data)[:, -1])))

    # test gbdt
    data = pd.read_csv("test.csv")[0: 1000]
    x = data[["feature0", "feature1", "feature2", "feature3", "feature4", "feature5", "feature6"]]
    y = data["clicked"]

    gbdt = DumbGbdt()
    gbdt.fit(x, y)

    from sklearn.metrics import roc_auc_score
    roc_auc_score(y, gbdt.predict(x))

