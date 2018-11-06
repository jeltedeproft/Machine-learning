__author__ = 'pvrancx'

import numpy as np
from scipy import optimize
from sklearn import metrics


#activation functions
def linear(s):
    return (s,1)


def logistic(s):
    logit = 1./(1+np.exp(-s))
    grad = logit * (1 - logit)
    return (logit,grad)


def tanh(s):
    logit = np.tanh(2*s/3)
    return (1.7159 *logit,1.7159*2/3*(1-logit**2)) 

#weight optimization functions
def flatten_weights(weightlist):
    res = np.array([])
    for w in weightlist:
        res = np.r_[res, w.flatten()]
    return res
    
def score_net(weights,net,X,y):
    net.load_weights(weights)
    return net.score(X,y)
    
def jac_net(weights,net,X,y):
    net.load_weights(weights)
    return flatten_weights(net._backprop(X,y))

#mlp class
class MultiLayerPerceptron(object):

    def __init__(self, nin,nhidden, nout, decay=0.0, niter=2000, actfun=logistic, outfun=logistic, scorefun='mse',alg='CG'):
        self._sizes = np.r_[nin, nhidden, nout]
        self._weights = []
        self._actfun = []
        for i in range(len(self._sizes)-1):
            win = self._sizes[i]+1
            wout = self._sizes[i+1]
            self._weights.append(np.random.normal(0.0, 0.05, (win, wout)))
            self._actfun.append(actfun)
        self._actfun[-1] = outfun
        self._decay = decay
        self._niter = niter
        self._alg = alg
        if scorefun == 'mse' or scorefun == 'ce':
            self._scorefun = scorefun
        else:
            raise Exception('scorefun should be either ce or mse')

    #forward propagation
    def predict(self,X):
        inp = X
        for w in range(len(self._weights)):
            #add bias column
            bias = np.ones(np.shape(inp)[0])
            inp = np.c_[bias, inp]
            #calculate activations
            act = np.dot(inp, self._weights[w])
            inp, grad = self._actfun[w](act)
        return inp

    #evaluate on test set
    def score(self,X,y):
        pred = self.predict(X)
        if self._scorefun == 'mse':
            return metrics.mean_squared_error(y,pred)
        elif self._scorefun == 'ce':
            #score = - np.sum( y*np.log(pred +1e-10)+(1-y)*np.log(1-pred+1e-10))
            return metrics.log_loss(y,np.c_[1-pred,pred])
        else:
            raise Exception('scorefun should be either ce or mse')

    #numerical gradient calculation
    def _numeric_grad(self,Xn,yn):
        eps=0.0001
        nw = []
        for w in range(len(self._weights)-1,-1,-1):
            grad = np.zeros(self._weights[w].shape)
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    val = self._weights[w][i,j]
                    self._weights[w][i,j]= val+eps
                    errPlus = np.sum((self.predict(Xn) - np.c_[yn])**2)/(2*Xn.shape[0])
                    self._weights[w][i,j]= val-eps
                    errMin= np.sum((self.predict(Xn) - np.c_[yn])**2)/(2*Xn.shape[0])
                    grad[i,j]= (errPlus-errMin)/ (2*eps)
                    self._weights[w][i,j]= val
            nw.append(grad)
        return nw
        
    def load_weights(self,weightarray):
        ind=0
        for w in range(len(self._weights)):
            wshape = self._weights[w].shape
            num_weights = np.prod(wshape)
            self._weights[w] = np.reshape(weightarray[ind:(ind+num_weights)],wshape)
            ind+=num_weights
    
    #backpropagation analytical gradient calculation
    def _backprop(self,X,y):
        inp = X
        acts = []
        grads = []

        #do forward prop
        for w in range(len(self._weights)):
            #add bias column
            bias = np.ones(np.shape(inp)[0])
            inp = np.c_[bias, inp]
            acts.append(inp)
            #calculate activations
            act = np.dot(inp, self._weights[w])
            inp, grad = self._actfun[w](act)
            grads.append(grad)


        #calculate output error

        output = inp
        
        #bacpropagate error
        delta = (output - np.c_[y])

        dw = []

        for w in range(len(self._weights)-1, -1, -1):
            #calculate layer gradient
            l_grad = np.dot(np.transpose(acts[w]), delta)
            #add regularization term
            l_grad += self._decay*np.r_[np.zeros((1,self._weights[w].shape[1])), self._weights[w][1:,:]]
            dw.append(l_grad / X.shape[0])
            #backpropagate error to next layer
            if w > 0 :
                #next layer error
                delta = np.dot(delta,np.transpose(self._weights[w]))
                #drop bias column
                delta = delta[:, 1:]
                delta *=  grads[w-1]

        dw.reverse()
        return dw

    #gradient optimization of weights
    def fit(self,X,y):
        optimize.minimize(score_net,flatten_weights(self._weights),jac=jac_net, args=(self,X,y), method=self._alg,options={'maxiter': self._niter,'gtol': 1e-5, 'disp': True})






