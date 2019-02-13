import mnist_loader
import ANNprog
import random
import numpy as np
import time

import matplotlib.pyplot as plt

def main():
    print "Effect of variation of Network Architecture and other variables on the classifier performance:"
    print "All variations done about the following : \n \
            Standard no of layers = 3\n \
            Standard no of nodes in hidden layers = 30\n \
            Standard no of iterations = 30\n \
            Standard mini batch size = 10\n \
            Standard learning rate = 3"

    print "Loading MNIST data...Please wait"
    
    train, test = mnist_loader.load_data_wrapper()

    print "Loading succesful"
    #standards:
    iterations=30
    mbl = 10
    n = 3.0
    size = [784, 30, 10]

    print "Varying the learning rate : "
    eta = [0.3, 1.0, 3.0]
    l=len(eta)
    accn = np.zeros(l)
    timen = np.zeros(l)
    tick1 = time.time()
 
    fig = plt.figure()
    fig1 = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig1.add_subplot(111)

    for i in xrange(len(eta)):
        clasify = ANNprog.ANN(size)
        accn[i],cost,accuracy=clasify.trainntest(train, iterations, mbl, eta[i], 0, test, need_cost=True)
        ax.plot(cost,label="$\eta$ = "+str(eta[i]))
        ax1.plot(accuracy,label="$\eta$ = "+str(eta[i]))
        tick2 = time.time()
        timen[i] = tick2-tick1 
        print "Learning rate = "+str(eta[i])+"\t accuracy = "+str(accn[i])+"\t Time = "+str(timen[i])
        tick1=tick2

    ax.set_xlim([0, iterations])
    ax.set_xlabel('No. of iterations')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

    ax1.set_xlim([0, iterations])
    ax1.set_xlabel('No. of iterations')
    ax1.set_ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.show()

    
    

if __name__ == '__main__':
    main()
