import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from scipy.optimize import basinhopping
from sklearn import preprocessing
from scipy.stats import truncnorm

import torch.nn as nn

def gen_d(a,b,gamma,H,K):
    return np.sin(gamma) / np.sqrt(H**2/a**2 + K**2/b**2 - 2*H*K*np.cos(gamma) / (a*b))

def gen_d_vector(a,b,gamma,H_max=10, K_max=10, noise=0):
    temp = np.zeros(2*H_max*K_max)
    temp2 = np.zeros(2*H_max*K_max)
    i=0
    #gaussian = np.random.normal(1, noise)
    if noise:
        t = truncnorm(-noise, noise)
    for H in range(-H_max, H_max):
        for K in range(0, H+1):
            if H == 0 and K == 0:
                continue
            d = gen_d(a, b, gamma, H, K)
            if noise:
                temp2[i] = d * (1- t.rvs(1)[0])
            else:
                temp2[i] = d
            temp[i] = d
            i+=1
    indices = np.argsort(temp[temp != 0])[:30]
    
    return np.array(temp[indices])

def gen_input(n):
    #a: 0.3-1.5
    #b: 0.5-2.5
    #gamma: 85-130
    #return [gen_d_vector(np.random.random(), np.random.random(), np.random.random()*50, 10, 10) for x in range(n)]
    return [[np.random.random()*1.2+0.3, np.random.random()*2+0.5, np.radians(85+np.random.random() * (130-85))] for x in range(n)]



if __name__ == '__main__':
    """
    yTr = np.array(gen_input(10000))
    xTr = np.array(list(map(lambda x: gen_d_vector(*x), yTr)))

    #xTr = np.array([[np.random.random()* 5] for x in range(1000)])
    #yTr = np.array([np.sin(x) for x in xTr])

    #model = MLPRegressor( hidden_layer_sizes=(150, 150)
    #                     ,batch_size=20,
    #                     activation='relu',
    #                      solver='adam', max_iter=100, tol=-1,verbose=True, learning_rate='constant', learning_rate_init=1e-3)
    #model.fit(xTr, yTr)

    scaler = preprocessing.StandardScaler().fit(xTr)
    xTr = scaler.transform(xTr)
    model2 = MLPRegressor( hidden_layer_sizes=(150, 150, 100)
                         ,batch_size=20,
                         activation='relu',
                           solver='adam', max_iter=100, tol=-1,verbose=True, learning_rate='constant', learning_rate_init=1e-3)
    model2.fit(xTr, yTr)

    yTe = np.array(gen_input(1000))
    xTe = np.array(list(map(lambda x: gen_d_vector(*x), yTe)))
    xTe_scaled = scaler.transform(xTe)
    #print(model.score(xTr, yTr))
    #print(model.score(xTe, yTe))

    #print(np.mean(np.abs(yTe - model.predict(xTe))))
    #print(np.mean(np.sum(np.abs(yTe - model.predict(xTe)), axis=1)))

    print(model2.score(xTr, yTr))
    print(model2.score(xTe, yTe))

    print(np.mean(np.abs(yTe - model2.predict(xTe))))
    print(np.mean(np.sum(np.abs(yTe - model2.predict(xTe)), axis=1)))
    print(np.mean(np.abs(yTe - model2.predict(xTe)), axis=0))

    #test: 0.65, 1.2, 111

    test_y = np.array([.65, 1.2, np.radians(111)]).reshape(1,-1)
    test_x =  gen_d_vector(*test_y[0])
    test_x = scaler.transform(test_x.reshape(1,-1))
    """
    xs = np.array([])
    ys = np.array([])
    a = 0.65
    b = 1.2
    g = np.radians(111)
    for M in range(-3, 3):
        for N in range(-3, 3):
            #if M == 0 and N == 0:
            #    continue
            xs = np.append(xs, M*a + N*b*np.cos(g))
            ys = np.append(ys, N*b*np.sin(g))

    plt.scatter(xs, ys)
    plt.scatter(xs+1, ys)
    plt.show()
