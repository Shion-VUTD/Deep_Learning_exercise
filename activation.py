import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    mask = (x<=0)
    x[mask] = 0
    return x
#まず、シグモイド関数で試してみる
#xは100次元*1000データ
def activate(f,hidden_layer_size = 5,node_num = 100, data_size = 1000,weight_init_std = 1):
    x = np.random.randn(data_size,node_num)  
    activations = {}
    for i in range(hidden_layer_size):
        if i != 0:
             x = activations[i-1]

        w = np.random.randn(node_num,node_num)*weight_init_std
        a = np.dot(x,w)
        z = f(a)
        activations[i] = z
    return activations    


#描画
#標準偏差1の場合
activations = activate(f = sigmoid,weight_init_std = 1)
for i,z in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(z.flatten(),30,range= (0,1))    

plt.show()    


#標準偏差0.01の場合
activations = activate(f=sigmoid,weight_init_std = 0.01)
for i,z in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(z.flatten(),30,range= (0,1))    

plt.show()    

#標準偏差Xavierの場合
node_num = 100
activations = activate(f = sigmoid,node_num = node_num, weight_init_std = 1/np.sqrt(node_num))
for i,z in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(z.flatten(),30,range= (0,1))    

plt.show()    

#relu関数の場合
#Xavier の初期値
node_num = 100
activations = activate(f = relu,node_num = node_num, weight_init_std = 1/np.sqrt(node_num))
for i,z in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(z.flatten(),30,range= (0,1))    

plt.show()    

#Heの初期値
node_num = 100
activations = activate(f = relu,node_num = node_num, weight_init_std = np.sqrt(2/node_num))
for i,z in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(z.flatten(),30,range= (0,1))    

plt.show()    