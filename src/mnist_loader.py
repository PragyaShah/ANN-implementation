#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    
    tr_d, va_d, te_d = load_data()
    tr_d=tr_d+va_d
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    #print training_inputs[1]
    training_inputs = training_inputs + [np.reshape(x, (784, 1)) for x in va_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_results = training_results + [vectorized_result(y) for y in va_d[1]]

    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
