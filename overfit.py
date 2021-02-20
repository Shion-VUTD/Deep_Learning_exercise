from test2 import TwoLayorNet,SGD
import numpy as np
import matplotlib.pyplot as plt
from deep_learning_scratch_for_exercise.dataset import mnist

(x_train,t_train),(x_test,t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)
x_train = x_train[:300]
t_train = t_train[:300]

network = TwoLayorNet(input_size=784,hidden_size=50,output_size=10)
train_size = x_train.shape[0]
batch_size = 50
epochs = 30
iters_per_epoch = train_size//batch_size
optimizer = SGD()
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    for iter in range(iters_per_epoch):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        #勾配
        loss = network.loss(x_batch,t_batch)
        grads = network.gradient(x_batch,t_batch)

        #SGD
        optimizer.upgrade(network.params,grads)

    #検証
    train_acc = network.accuracy(x_train,t_train)
    test_acc = network.accuracy(x_test,t_test)
    print('epoch:',epoch,'train_accuracy:',train_acc,'test_accuracy:',test_acc)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

plt.plot(train_acc_list,label = 'train')
plt.plot(test_acc_list,label = 'test')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()    
 

