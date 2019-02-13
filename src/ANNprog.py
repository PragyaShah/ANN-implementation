import random
import numpy as np



class ANN(object):

    def __init__(self, lsize):

        self.layers = len(lsize)#no of layers
        self.lsize = lsize#size of each layer
        self.b = [np.random.randn(i, 1) for i in lsize[1:]]
        self.w = [np.random.randn(j, i) for i, j in zip(lsize[:-1], lsize[1:])]
        #print "Neural network creation successful! "+str(self.layers)+" layers created"

        
    def trainntest(self, train, rounds, batchlen, lrate, prnt, test, need_cost=False):

        cost,accuracy = [], []
        len_train = len(train)
        len_test = len(test)
        maxacc = 0.0
        if prnt:
            print "Training the network...Please wait"
        for i in xrange(rounds):
            random.shuffle(train)
            batcharr = [train[k:k+batchlen] for k in xrange(0, len_train, batchlen)]
            for batch in batcharr:
                self.update(batch, lrate)
            if prnt:
                print " Round "+str(i)+" completed"
            
            if need_cost:
                costs = self.total_cost(test, lrate, convert=True)
                cost.append(costs)
                #print "Cost on test data: {}".format(costs)

            corclass = self.perform(test)
            acc = corclass*100.0/len_test

            accuracy.append(acc)
            if prnt:
                print str(corclass)+" out of "+str(len_test)+" images classified correctly"
            if (maxacc<acc):
                maxacc = acc
        if prnt:
            print "Training complete! \n   Maximum accuracy = "+str(maxacc)
        return (maxacc,cost,accuracy)

    def update(self, batch, lrate):
        sumdel_b = [np.zeros(i.shape) for i in self.b]
        sumdel_w = [np.zeros(i.shape) for i in self.w]
        for x, y in batch:
            delta_b, delta_w = self.prop(x,y)
            sumdel_b = [prevsmb+delb for prevsmb, delb in zip(sumdel_b, delta_b)]
            sumdel_w = [prevsmw+delw for prevsmw, delw in zip(sumdel_w, delta_w)]
        self.b = [orgb - (lrate/len(batch))*sumdelb for orgb, sumdelb in zip(self.b, sumdel_b)]
        self.w = [orgw - (lrate/len(batch))*sumdelw for orgw, sumdelw in zip(self.w, sumdel_w)]

    def prop(self, x, y):
        del_b = [np.zeros(i.shape) for i in self.b]
        del_w = [np.zeros(i.shape) for i in self.w]

        inp = x
        layeroutputs = [x]#as first layer simply fans out inputs
        layerwiseeta = []

        for b, w in zip(self.b, self.w):
            eta = np. dot(w, inp)+b
            layerwiseeta.append(eta)
            inp = actfunct(eta)
            layeroutputs.append(inp)

        delta = (layeroutputs[-1]-y)*dervactfunct(layerwiseeta[-1])
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, layeroutputs[-2].transpose())

        for l in xrange(2, self.layers):
            eta = layerwiseeta[-l]
            delta = np.dot(self.w[-l+1].transpose(), delta)*dervactfunct(eta)
            del_b[-l] = delta
            del_w[-l] = np.dot(delta, layeroutputs[-1-l].transpose())
        return (del_b, del_w)

    def perform (self, test):
        corr = 0
        #res = [(np.argmax(self.outp(x)), y) for (x, y) in test]
        for (x,y) in test:
            if (np.argmax(self.outp(x))==y):
                corr+=1
        return corr
    
    def outp(self, x):
        for b, w in zip(self.b, self.w):
            x = actfunct(np.dot(w, x)+b)
        return x

    def total_cost(self, test, lrate, convert=False):
        
        cost = 0.0
        for x, y in test:
            a = self.outp(x)
            if convert: y = vectorized_result(y)
            cost += fn(a, y)
        
        return cost/len(test)

     

def actfunct(x):
    return 1.0/(1.0+np.exp(-x))#we are implementing sigmoid

def dervactfunct(x):
    return actfunct(x)*(1-actfunct(x))#we are implementing sigmoid


def vectorized_result(j):
         e = np.zeros((10, 1))
         e[j] = 1.0
         return e

def fn(a, y):
        
        return 0.5*np.linalg.norm(a-y)**2

