import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

class Data(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def split(self, percentage):  # 0.8
        X = self.dataset[:, 0][::, None]
        y = self.dataset[:, 1][::, None]
        permuted_idxs = np.random.permutation(X.shape[0])
        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]
        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]
        X_train = X[train_idxs]
        X_test = X[test_idxs]
        y_train = y[train_idxs]
        y_test = y[test_idxs]
        return X_train, X_test, y_train, y_test

def Gradiente_Descendiente(X,Y,alpha,epochs):
    n,m = np.shape(Y)
    w = np.random.uniform(0, 1, (np.shape(X)[1], 1)) #Se puede cambiar
    for i in range(epochs):
        aux1 = (Y - np.matmul(X, w)) * X
        grad =-2 * np.sum(aux1, axis=0) / n
        #print(aux1)
        w = w - alpha * grad[::,None]
    return w

def Gradiente_Estocastico(X,Y,alpha,epochs):
    n = np.shape(Y)[0]
    w = np.random.uniform(0, 1, (np.shape(X)[1], 1)) #Se puede cambiar
    for i in range(epochs):
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        Y = Y[idx]
        for i in range(n):
            aux1 = (Y[i] - np.matmul(X[i],w)) * X[i]
            grad = -2 * aux1 / n
            w = w - alpha * grad[::,None]
        #print(w)
    return w

def Gradiente_MiniBatch(X,Y,alpha,epochs):
    b = 16 #Se divide en 16 batches
    n = np.shape(Y)[0]
    w = np.random.uniform(0, 1, (np.shape(X)[1], 1)) #Se puede cambiar
    batch = int(n/b)
    for i in range(epochs):
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        Y = Y[idx]
        for i in range(0,n,batch):
            aux1 = (Y[i:i+batch-1] - X[i:i+batch-1]* w) * X[i:i+batch-1]
            grad = -2 * np.sum(aux1,axis=0) / n
            w = w - alpha * grad[::,None]
        #print(w)
    return w

my_data = genfromtxt('income.data.csv', delimiter=',')
Datos = Data(my_data[:,1:])
X_train, X_test, Y_train, Y_test = Datos.split(0.8)
alpha = 0.01 #Learning Rate
epochs = 100
f = np.sin(X_train)
X1 = np.append(X_train,np.ones((np.shape(f)[0],1)),axis=1)
X2 = np.append(X_train**2,X1,axis=1)
X3 = np.append(X_train**3,X2,axis=1)
X4 = np.append(X_train**4,X3,axis=1)
X5 = np.append(X_train**5,X4,axis=1)
X6 = np.append(X_train**6,X5,axis=1)
X7 = np.append(X_train**7,X6,axis=1)
X8 = np.append(X_train**8,X7,axis=1)
X9 = np.append(X_train**9,X8,axis=1)
X10 = np.append(X_train**10,X9,axis=1)
W1 = Gradiente_Estocastico(X1,f,alpha,epochs)
W2 = Gradiente_Estocastico(X2,f,alpha,epochs)

#print(W2)
plt.plot(X_train,f,'bs')
plt.plot(X_train,np.matmul(X1,W1),'rs')
plt.plot(X_train,np.matmul(X2,W2),'gs')
plt.show()