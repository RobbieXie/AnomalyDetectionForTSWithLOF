import numpy
import matplotlib.pyplot as plt

from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, softdtw_barycenter
from tslearn.datasets import CachedDatasets

numpy.random.seed(0)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X = X_train[y_train == 2]

plt.figure()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.subplot(3, 1, 1)
for ts in X:
    plt.plot(ts.ravel(), "k-", alpha=.2)
plt.plot(euclidean_barycenter(X).ravel(), "r-", linewidth=2)
plt.title("算数平均序列求解")

plt.subplot(3, 1, 2)
sdtw_bar = softdtw_barycenter(X, gamma=1., max_iter=100)
for ts in X:
    plt.plot(ts.ravel(), "k-", alpha=.2)
plt.plot(sdtw_bar.ravel(), "r-", linewidth=2)
plt.title("DBA平均序列求解")

plt.tight_layout()
plt.show()