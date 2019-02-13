import mnist_loader
import ANNprog
import random
import numpy as np
import matplotlib.pyplot as plt

def main():
    print "Network Architecture:"
    
    print "Number of layers in architecture :"
    l = int(raw_input())
    
    print "Enter number of nodes in each layer"
    lsize = map (int,(raw_input().strip().split()))

    clasify = ANNprog.ANN(lsize)

    print "Loading MNIST data...Please wait"
    
    train, test = mnist_loader.load_data_wrapper()

    print "Loading succesful"

    print "Classifier to be trained for how many iterations?"
    i = int(raw_input())
    print "Enter mini batch length"
    mbl = int(raw_input())
    print "Learning rate of classifier"
    n = float(raw_input())

    acc,cost,accuracy=clasify.trainntest(train, i, mbl, n, 1, test,need_cost=True)
    
    plt.plot(cost)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.xlim(0,i)
    plt.savefig('cost plot.png') 
    plt.show()


    plt.plot(accuracy)
    plt.xlabel('No. of iterations')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0,i)
    plt.savefig('Accuracy plot.png') 
    plt.show()
    

if __name__ == '__main__':
    main()
