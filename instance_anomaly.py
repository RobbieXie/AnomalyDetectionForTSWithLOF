import numpy as np
from matplotlib import pyplot as plt
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, softdtw_barycenter
from tslearn.datasets import CachedDatasets
import math
from queue import Queue,PriorityQueue
from dtaidistance import dtw


# Calculate the distance of each point
def getdisance(data, datalen):
    # print data
    dismatrix = [[0 for i in range(datalen)] for i in range(datalen)]

    for i in range(datalen):
        for j in range(datalen):
            use_dtw = 0
            dis = 0
            if use_dtw == 0:
                for l in range(len(data[0])):
                    dis = dis + (data[i][l] - data[j][l]) * (data[i][l] - data[j][l])
                dis = math.sqrt(dis)
            else:
                dis = dtw.distance(data[i],data[j])
            dismatrix[i][j] = dis
            dismatrix[j][i] = dis
    return dismatrix


# Calculate the K distance of each point
def getk_distance(matrix, datalen, k):
    kdis = [0 for i in range(datalen)]
    for i in range(datalen):
        pq = PriorityQueue(k)
        for j in range(datalen):
            if i != j:
                if pq.full() == False:
                    pq.put(matrix[i][j] * -1)
                else:
                    kdis[i] = pq.get()
                    if matrix[i][j] * -1 > kdis[i]:
                        pq.put(matrix[i][j] * -1)
                    else:
                        pq.put(kdis[i])
        kdis[i] = pq.get() * -1
    print("kids:", kdis)
    return kdis


# Calculate reachable disdance of each point
def getreach_distance(matrix, datalen, kdis):
    reachdis_matrix = [[0 for i in range(datalen)] for i in range(datalen)]
    for i in range(datalen):
        for j in range(datalen):
            if i == j:
                reachdis_matrix[i][j] = 0
            else:
                if matrix[i][j] > kdis[j]:
                    reachdis_matrix[i][j] = matrix[i][j]
                else:
                    reachdis_matrix[i][j] = kdis[j]
    # print("reachdis_matrix:"reachdis_matrix)
    return reachdis_matrix


# Calculate local reachable density of each point
def getlrd(reachdis_matrix, matrix, datalen, minpts):
    lrd = [0 for i in range(datalen)]
    for i in range(datalen):
        lrdpq = PriorityQueue(minpts)
        for j in range(datalen):
            if i != j:
                if lrdpq.full() == False:
                    lrdpq.put([matrix[i][j] * -1, j])
                else:
                    temp = lrdpq.get()
                    if matrix[i][j] * -1 > temp[0]:
                        lrdpq.put([matrix[i][j] * -1, j])
                    else:
                        lrdpq.put(temp)
        while not lrdpq.empty():
            temp = lrdpq.get()
            lrd[i] = lrd[i] + reachdis_matrix[i][temp[1]]
        lrd[i] = minpts / (lrd[i] + 0.001)

    print("lrd:", lrd)
    return lrd


# Calculate LOF of each point
def getlof(data, k, minpts):
    datalen = len(data)
    dismatrix = getdisance(data, datalen)
    kdis = getk_distance(dismatrix, datalen, k)
    reach_mat = getreach_distance(dismatrix, datalen, kdis)
    lrd = getlrd(reach_mat, dismatrix, datalen, minpts)

    lof = [0 for i in range(datalen)]
    for i in range(datalen):
        lofpq = PriorityQueue(minpts)
        for j in range(datalen):
            if i != j:
                if lofpq.full() == False:
                    lofpq.put([dismatrix[i][j] * -1, j])
                else:
                    temp = lofpq.get()
                    if dismatrix[i][j] * -1 > temp[0]:
                        lofpq.put([dismatrix[i][j] * -1, j])
                    else:
                        lofpq.put(temp)
        while not lofpq.empty():
            temp = lofpq.get()
            lof[i] = lof[i] + lrd[temp[1]]
        lof[i] = (lof[i] / minpts) / lrd[i]

    print("lof:", lof)

    return lof


def main():
    data_path = r'data.txt'
    data = np.loadtxt(data_path)
    data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11 = data.T
    X=[data1, data2, data4, data5, data6, data7, data8, data9, data10, data11]
    X1 = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
    X2 = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data11]

    K = 3
    MinPts = K
    # test data
    psdata = [[55, 88, 40, 62, 75],
              [40, 85, 45, 65, 75],
              [55, 80, 40, 58, 75],
              [45, 88, 42, 62, 70],
              [55, 82, 38, 66, 78],
              [55, 50, 99, 99, 20],
              [58, 70, 40, 60, 80],
              [0, 0, 0, 0, 0]]
    print("psdata len:", len(X))

    pslof = getlof(X, K, MinPts)
    # pslof1 = getlof(X1, K, MinPts)
    # pslof2 = getlof(X2, K, MinPts)

    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.subplot(2, 1, 1)
    plt.ylim(0, 100)
    for i in range(len(X)):
        ts = X[i]
        if i == 8:
            plt.plot(ts.ravel(), alpha=.5, label="网络异常")
        elif i == 9:
            plt.plot(ts.ravel(), alpha=.5, label="CPU加压异常")
        else:
            plt.plot(ts.ravel(), alpha=.5)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Utilization')
    plt.title("各Sprout实例的CPU使用率负载曲线")

    # plt.subplot(4, 1, 4)
    # plt.ylim(0, 100)
    # for i in range(len(X1)):
    #     ts = X1[i]
    #     is_anomaly = (pslof1[i] > 2)
    #     if is_anomaly:
    #         plt.plot(ts.ravel(), "r-", linewidth=2)
    #     else:
    #         plt.plot(ts.ravel(), "k-", alpha=.4)
    # plt.title("网络异常检测结果")
    #
    # plt.subplot(4, 1, 3)
    # plt.ylim(0, 100)
    # for i in range(len(X2)):
    #     ts = X2[i]
    #     is_anomaly = (pslof2[i] > 2)
    #     if is_anomaly:
    #         plt.plot(ts.ravel(), "r-", linewidth=2)
    #     else:
    #         plt.plot(ts.ravel(), "k-", alpha=.4)

    # plt.title("CPU异常检测结果")

    plt.subplot(2, 1, 2)
    plt.ylim(0, 100)

    label_anomaly = 0
    label_nomal = 0
    for i in range(len(X)):
        ts = X[i]
        is_anomaly = (pslof[i] > 2)
        if is_anomaly:
            if label_anomaly == 0:
                label_anomaly = 1
                plt.plot(ts.ravel(), "r-", linewidth=1, label="异常实例")
            else:
                plt.plot(ts.ravel(), "r-", linewidth=1)
        else:
            if label_nomal == 0:
                label_nomal = 1
                plt.plot(ts.ravel(), "k-", alpha=.5, label="正常实例")
            else:
                plt.plot(ts.ravel(), "k-", alpha=.5)

    plt.title("异常检测结果")

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Utilization')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()