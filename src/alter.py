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
            Standard no of iterations = 30\n\
            Standard mini batch size = 10\n\
            Standard learning rate = 3"

    print "Loading MNIST data...Please wait"
    
    train, test = mnist_loader.load_data_wrapper()

    print "Loading succesful"
    #standards:
    iterations=30
    mbl = 10
    n = 3.0
    size = [784, 30, 10]

    print "Varying the number of layers in architecture : "
    lay = [2, 3, 4]
    lsize = [[784, 10],[784, 30, 10],[784, 50, 30, 10]]
    l=len(lay)
    accl = np.zeros(l)
    timel = np.zeros(l)
    tick1 = time.time()

    for i in xrange(len(lsize)):
        clasify = ANNprog.ANN(lsize[i])
        acci[i],cost,accuracy=clasify.trainntest(train, iterate, mbl, n, 0, test,need_cost=True)
        tick2 = time.time()
        timel[i] = tick2-tick1
        print "Layers = "+str(lay[i])+"\t Accuracy = "+str(accl[i])+"\t Time = "+str(timel[i])
        tick1=tick2
  

    print "Varying the learning rate : "
    eta = [0.3, 1.0, 3.0, 10.0, 50.0, 100.0]
    l=len(eta)
    accn = np.zeros(l)
    timen = np.zeros(l)
    tick1 = time.time()
 
    

    for i in xrange(len(eta)):
        clasify = ANNprog.ANN(size)
        accn[i],cost,accuracy=clasify.trainntest(train, iterations, mbl, eta[i], 0, test,need_cost=True)
        tick2 = time.time()
        timen[i] = tick2-tick1 
        print "Learning rate = "+str(eta[i])+"\t accuracy = "+str(accn[i])+"\t Time = "+str(timen[i])
        tick1=tick2

    
    print "Varying the batch size : "
    batch = [5, 10, 50, 100, 1000]
    l=len(batch)
    accb = np.zeros(l)
    timeb = np.zeros(l)
    tick1 = time.time()
    for i in xrange(len(batch)):
        clasify = ANNprog.ANN(size)
        accb[i]=clasify.trainntest(train, iterations, batch[i], n, 0, test)
        tick2 = time.time()
        timeb[i] = tick2-tick1
        print "Mini batch size = "+str(batch[i])+"\t accuracy = "+str(accb[i])+"\t Time = "+str(timeb[i])
        tick1=tick2
        
    print "Varying the number of iterations : "
    iterate = [5, 15, 30, 50, 70]
    l=len(iterate)
    acci = np.zeros(l)

    timei = np.zeros(l)
    tick1 = time.time()
    
    for i in xrange(len(iterate)):
        clasify = ANNprog.ANN(size)
        acci[i],cost,accuracy=clasify.trainntest(train, iterate[i], mbl, n, 0, test,need_cost=True)
        ax.plot(cost,label="Iteration = "+str(iterate[i]))
        ax1.plot(accuracy,label="Iteration = "+str(iterate[i]))
        tick2 = time.time()
        timei[i] = tick2-tick1
        print "No of iterations = "+str(iterate[i])+"\t accuracy = "+str(acci[i])+"\t Time = "+str(timei[i])
        tick1=tick2
    
    print "Varying the number of nodes in the hidden layer : "
    hidsize = [20, 30, 50, 100]
    l=len(hidsize)
    acch = np.zeros(l)
    timeh = np.zeros(l)
    tick1 = time.time()
    for i in xrange(len(hidsize)):
        clasify = ANNprog.ANN([784, hidsize[i], 10])
        acch[i],cost,accuracy=clasify.trainntest(train, iterations, mbl, n, 0, test,need_cost=True)
        tick2 = time.time()
        timeh[i] = tick2-tick1
        print "No of nodes in hidden layer = "+str(hidsize[i])+"\t Accuracy = "+str(acch[i])+"\t Time = "+str(timeh[i])
        tick1=tick2
    
    
    
        
    
        
    

   
    #graphs
        
    

if __name__ == '__main__':
    main()
