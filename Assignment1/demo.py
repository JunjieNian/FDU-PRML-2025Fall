import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. 加载简单数据（这里用鸢尾花数据集的前两个特征，方便可视化）
iris = load_iris()
X = iris.data[:, :2]  # 只取前两个特征
y = iris.target

# 2. 训练决策树
clf = DecisionTreeClassifier(max_depth=None, random_state=42)
clf.fit(X, y)

# 3. 打印树的深度和叶子数
print("树的最大深度:", clf.get_depth())
print("树的叶子节点数:", clf.get_n_leaves())

# 4. 可视化决策树
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
plt.show()
