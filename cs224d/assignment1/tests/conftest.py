import numpy as np
import random


def assert_close(first, second, tolerance=1e-4, norm=1.0):
    print first, second
    if type(first) is np.ndarray and len(first.shape) > 0:
        diff = np.abs(first - second) / np.maximum(norm, first, second)
        violated = diff > tolerance
        if violated.any():
            print "input data: checking \n%s vs \n%s" % (first, second)
            print "diff\n", diff
            #print "gradient violated at indices %s" % violated
        assert not violated.any()

    else:
        violated = abs(first - second) / max(norm, first, second) > tolerance
        print "input data: checking \n%s vs \n%s" % (first, second)
        assert not violated  

      
def empirical_grad(f, x, step=1e-4, verbose=False):

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx = f(x)[0] # Evaluate function value at original point

    numgrad = np.zeros_like(x)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        if verbose:
            print "Gradient check at dimension %s" % str(ix)
        
        x[ix] += 0.5 * step
        random.setstate(rndstate)
        f2 = f(x)[0]
        x[ix] -= step
        random.setstate(rndstate)
        f1 = f(x)[0]
        numgrad[ix] = (f2 - f1) / step
        x[ix] += 0.5 * step
        it.iternext() # Step to next dimension
    return numgrad


