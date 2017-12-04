import numpy as np
import random
import threading
import time
from multiprocessing import Process, Queue, freeze_support, set_start_method


def matrix_thread(A, B, C, K, N, n1, n2):
    for i in range(n1, n2):
        for j in range(0, K):
            for k in range(0, N):
                C[i, j] += A[i, k] * B[k, j]


def matrix_multy(q, A, B, K, N, n1, n2):
    C_new = 0
    list_of_C = []
    for i in range(n1, n2):
        for j in range(0, K):
            for l in range(0, N):
                C_new += A[i, l] * B[l, j]
            list_of_C.append([C_new, i, j])

    q.put(list_of_C)


if __name__ == '__main__':

    M = 1000
    N = 1000
    K = 100
    A = np.zeros((M, N))
    B = np.zeros((N, K))
    C = np.zeros((M, K))

    for i in range(0, M):
        for j in range(0, N):
            A[i][j] = random.randrange(-10, 10)

    for i in range(0, N):
        for j in range(0, K):
            B[i][j] = random.randrange(-10, 10)

    list_of_threading = []
    list_of_Process = []
    list_of_Queue = []

    freeze_support()
    set_start_method('spawn')

    for i in range(0, 4):
        list_of_threading.append(threading.Thread(target=matrix_thread, args=(A, B, C, K, N, int(i * M / 4), int(i * M / 4 + M / 4))))
        q = Queue()
        list_of_Queue.append(q)
        list_of_Process.append(Process(target=matrix_multy, args=(q, A, B, K, N, int(i * M / 4), int(i * M / 4 + M / 4))))

    time_start = time.time()
    if True:
        matrix_thread(A, B, C, K, N, 0, M)
        print(time.time() - time_start)

    threading = True
    time_start = time.time()
    if threading:
        for elem in list_of_threading:
            elem.start()
        for elem in list_of_threading:
            elem.join()
    else:
        for elem in list_of_Process:
            elem.start()

        for elem in list_of_Queue:
            data = elem.get()
            for tt in data:
                C[tt[1], tt[2]] = tt[0]

    print(time.time() - time_start)
