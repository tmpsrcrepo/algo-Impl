import numpy as np
import random

def getFunc(f, x, rndstate):
    random.setstate(rndstate)
    return f(x)

def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    fx, grad = getFunc(f, x, rndstate)
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        #f(x+h) - f(x-h) / 2h
        v = x[ix]
        x[ix] += h
        f1, _ = getFunc(f, x, rndstate)
        x[ix] = v - h
        f0, _ = getFunc(f, x, rndstate)
        numgrad = (f1-f0)/(2*h)
        x[ix] = v

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"
