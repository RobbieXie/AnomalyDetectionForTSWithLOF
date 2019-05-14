import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
print(X_train)
print(y_train)
X_train = X_train[y_train < 4]  # Keep first 3 classes
numpy.random.shuffle(X_train)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)  # Make time series shorter
sz = X_train.shape[1]
print(sz)
print(X_train.shape)

xie_1 = []
xie_2 = numpy.array([X_train[1]])
xie_3 = numpy.array([X_train[3]])
xie_4 = numpy.array([X_train[2]])
print(xie_2)
print(numpy.reshape(X_train[0],(40)))
y1 = [1]
y2 = [2]
y3 = [3]
y4 = [4]


for i in range(50):
    if y_train[i] == 1:
        xie_1 = xie_1 + numpy.reshape(X_train[i],(40))
        y1.append(1)
    # if y_train[i] == 2:
    #     xie_2 += X_train[i]
    # if y_train[i] == 3:
    #     xie_3 += X_train[i]
    # if y_train[i] == 4:
    #     xie_4 += X_train[i]

print(y1)




# Euclidean k-means
# print("Euclidean k-means")
# km = TimeSeriesKMeans(n_clusters=2, verbose=True, random_state=seed)
# y_pred = km.fit_predict(X_train)
#
# plt.figure()
#
# for yi in range(2):
#     plt.subplot(3, 3, yi + 1)
#     for xx in X_train[y_pred == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     if yi == 1:
#         plt.title("Euclidean $k$-means")
#
# # DBA-k-means
# print("DBA k-means")
# dba_km = TimeSeriesKMeans(n_clusters=2, n_init=2, metric="dtw", verbose=True, max_iter_barycenter=10, random_state=seed)
# y_pred = dba_km.fit_predict(X_train)
#
# for yi in range(2):
#     plt.subplot(3, 3, 4 + yi)
#     for xx in X_train[y_pred == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     if yi == 1:
#         plt.title("DBA $k$-means")

# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=2, metric="softdtw", metric_params={"gamma_sdtw": .01},
                           verbose=True, random_state=seed)
y_pred = sdtw_km.fit_predict(xie_1)

for yi in range(3):
    plt.subplot(3, 3, 7 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")


plt.subplot(3, 3, 3)
for xx in X_train:
    plt.plot(xx.ravel(), "g-", alpha=.2)

plt.tight_layout()
plt.show()