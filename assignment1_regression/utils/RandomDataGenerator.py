import numpy as np

class Data:pass

E = lambda x: np.insert(x, 0, 1, axis=1)


"""
THIS IS THE ANALYTICAL APPROACH FOR SOLVING LINEAR EQUATION
TRY TO UNDERSTAND THIS FUNCTION by playing with this with some random data, and theory behind it
We will use this only for our smaller dataset with only 1 feature in x and 1 ydim,
"""

def analytical(x, y):
    x = E(x)
    theta_best = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return x.dot(theta_best)

"""
RANDOM generator generates data randomly
here xdim is dimension of x, i.e, number of features in x
and ydim is for label

n refers to number of data sample which is further splitted into train-test-val set
"""

def getrandxy(n = 100, xdim = 1, ydim = 1):
    x = np.random.rand(n, xdim)
    theta = np.random.rand(xdim + 1, ydim)
    expected_theta = np.random.randint(-100, 100, (xdim + 1, ydim))
    y = E(x).dot(expected_theta) + np.random.randint(-20, 20, (n, ydim))

    r = int(n*0.2)
    data = Data()

    data.x = x
    data.y = y

    data.testx, data.testy = x[0:r,:], y[0:r,:]
    data.valx , data.valy = x[r:r*2,:], y[r:r*2,:]
    data.trainx, data.trainy = x[r*2:,:],y[r*2:,:]

    data.theta = theta
    data.expected_theta = expected_theta

    return data

