import autograd.numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numba import njit, prange
import numba
import time
from autograd import grad, elementwise_grad, jacobian


def feed_forward(X, Y, W0, W1, W2, W3, W4, return_loss=True):
    ''' simple 4-layer feed forward neural net. 
    Layers are hardcoded but can easily be modified. 

    Parameters
    ----------
    X : is a 2-D array with shape (m, n) where m is the number of input variables
        and n the number of dimensions. Here for a 2-D problem we have n = 2 + 1
        because of the bias. Last row of the input array has to be all "1".
    Y : Target vector with shape (m, n) where m is the number of corresponding 
        output values and n is only one)

    W0...W4 : pre-created weight arrays. Used dimensions need to allow for matrix multiplications

    return_loss : flag to switch between the loss and the actual result  

    Returns
    -------

    loss: l2 error metric loss
    l5 : value of the very last layer for evaluation


    '''

    def ReLU(x, deriv=False):
        if(deriv == True):
            return 1 * (x > 0)
        return x * (x > 0)

    def lin(x, deriv=False):
        if(deriv == True):
            return 1
        return x

    def sigmoid(x, deriv=False):

        if(deriv == True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    def leakyReLU(x, deriv=False):
        if(deriv == True):
            return np.where(x > 0, 1, 0.01)
        return x * (x > 0) + 0.01 * x * (x < 0)

    l0 = X
    l1 = ReLU(np.dot(l0, W0))
    l2 = ReLU(np.dot(l1, W1))
    l3 = ReLU(np.dot(l2, W2))
    l4 = ReLU(np.dot(l3, W3))
    l5 = lin(np.dot(l4, W4))

    if return_loss == True:
        loss = (Y-l5)**2  # simple quadratic loss
        return loss
    else:
        return l5


# define input array
X = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0.5, 0.5],
              [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5], [0, 0], [0, 0]])
X = np.hstack((X, np.ones((np.shape(X)[0], 1))))  # add bias

# define output array
Y = np.array([[0.3], [0.3], [0.5], [0.6], [0.5],
              [0.5], [0.5], [0.7], [0.8], [0.85]])

# create weight arrays with random initialization. Zero centered
W0 = (2*np.random.random((2+1, 128)) - 1)*0.2
W1 = (2*np.random.random((128, 64)) - 1)*0.2
W2 = (2*np.random.random((64, 32)) - 1)*0.2
W3 = (2*np.random.random((32, 16)) - 1)*0.2
W4 = (2*np.random.random((16, 1)) - 1)*0.2

# optimizers like adam and rmp-prob need more array space to save values
W0_momentum = np.zeros_like(W0)
W1_momentum = np.zeros_like(W1)
W2_momentum = np.zeros_like(W2)
W3_momentum = np.zeros_like(W3)
W4_momentum = np.zeros_like(W4)

W0_history = np.zeros_like(W0)
W1_history = np.zeros_like(W1)
W2_history = np.zeros_like(W2)
W3_history = np.zeros_like(W3)
W4_history = np.zeros_like(W4)


# set the z axis limits so they aren't recalculated each frame.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# start main iteration loop
wframe = None
for i in range(0, 10000):
    print("Iteration {}".format(i))

    if wframe:
        ax.collections.remove(wframe)
        ax.collections.remove(scatter)

    a = 0.001
    eps = 1*10**-8
    beta1 = 0.9
    beta2 = 0.95
    optimizer = "adam"

    W0_grad = elementwise_grad(feed_forward, 2)(X, Y, W0, W1, W2, W3, W4)
    W1_grad = elementwise_grad(feed_forward, 3)(X, Y, W0, W1, W2, W3, W4)
    W2_grad = elementwise_grad(feed_forward, 4)(X, Y, W0, W1, W2, W3, W4)
    W3_grad = elementwise_grad(feed_forward, 5)(X, Y, W0, W1, W2, W3, W4)
    W4_grad = elementwise_grad(feed_forward, 6)(X, Y, W0, W1, W2, W3, W4)

    if optimizer == "rmsprop":
        W0_history = W0_history * beta1 + (1 - beta1) * W0_grad**2
        W1_history = W1_history * beta1 + (1 - beta1) * W1_grad**2
        W2_history = W2_history * beta1 + (1 - beta1) * W2_grad**2
        W3_history = W3_history * beta1 + (1 - beta1) * W3_grad**2

        W0 = W0 - (a/np.sqrt(W0_history + eps)) * W0_grad
        W1 = W1 - (a/np.sqrt(W1_history + eps)) * W1_grad
        W2 = W2 - (a/np.sqrt(W2_history + eps)) * W2_grad
        W3 = W3 - (a/np.sqrt(W3_history + eps)) * W3_grad

    if optimizer == "adam":
        W0_momentum = beta1 * W0_momentum + (1 - beta1) * W0_grad
        W1_momentum = beta1 * W1_momentum + (1 - beta1) * W1_grad
        W2_momentum = beta1 * W2_momentum + (1 - beta1) * W2_grad
        W3_momentum = beta1 * W3_momentum + (1 - beta1) * W3_grad
        W4_momentum = beta1 * W4_momentum + (1 - beta1) * W4_grad

        W0_history = W0_history * beta2 + (1 - beta2) * W0_grad**2
        W1_history = W1_history * beta2 + (1 - beta2) * W1_grad**2
        W2_history = W2_history * beta2 + (1 - beta2) * W2_grad**2
        W3_history = W3_history * beta2 + (1 - beta2) * W3_grad**2
        W4_history = W4_history * beta2 + (1 - beta2) * W4_grad**2

        W0 = W0 - (a / np.sqrt(W0_history + eps)) * W0_momentum
        W1 = W1 - (a / np.sqrt(W1_history + eps)) * W1_momentum
        W2 = W2 - (a / np.sqrt(W2_history + eps)) * W2_momentum
        W3 = W3 - (a / np.sqrt(W3_history + eps)) * W3_momentum
        W4 = W4 - (a / np.sqrt(W4_history + eps)) * W4_momentum

    if optimizer == "sgd":
        W0 = W0 - a * W0_grad
        W1 = W1 - a * W1_grad
        W2 = W2 - a * W2_grad
        W3 = W3 - a * W3_grad

    # set up mesh grid
    xs = np.linspace(-1, 1, 40)
    ys = np.linspace(-1, 1, 40)
    Xs, Ys = np.meshgrid(xs, ys)

    # convert grid to neural net input dimensions
    Xnet = np.hstack((Xs.reshape((40*40, 1)), Ys.reshape((40*40, 1))))
    Xnet = np.hstack((Xnet, np.ones((np.shape(Xnet)[0], 1))))  # add bias

    # evaluate net for current results. Return shape to grid-like structure
    Znet = feed_forward(Xnet, Y, W0, W1, W2, W3, W4, return_loss=False)
    Znet = Znet.reshape((40, 40))

    # plot results
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_zlim3d(0, 1)
    wframe = ax.plot_surface(Xs, Ys, Znet, cmap="CMRmap")
    scatter = ax.scatter(X[:, 0], X[:, 1], Y, c="b", marker="o")
    plt.pause(.00001)
