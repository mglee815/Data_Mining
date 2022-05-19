import numpy as np
import P1_1
from tqdm import tqdm


#Sigmoid
class Sigmoid:
    def __init__(self):
        self.y = None
    
    #case1 : online
    #case2 : batch - 아직 구현 못함
    def forward(self, X):
        self.y = 1 / (1+np.exp(-X))
        return self.y
    
    #case1 : online
    #case2 : batch - 구현X
    def backward(self, dout):
        dout_lst = dout * self.y * (1.0 - self.y)
        return dout_lst
    
class cross_entropy:
    def __init__(self):
        self.cee_out = None
    
    def forawrd(self, y, sig_out):
        self.y = y.copy()
        self.sig_out = sig_out.copy()
        self.cee_out = np.mean( - (  (self.y * np.log(self.sig_out) ) + ( (1 - self.y) * np.log(1 - self.sig_out) ) ) )
        print(self.cee_out)
        return np.mean(self.cee_out)

        
    def backward(self):
        return (self.sig_out - self.y) / self.sig_out * (1 - self.sig_out)
    
    

class Affine:
    def __init__(self, w, b, lr = 1):
        self.x = None # X = [x1, x2] {nd.array}
        self.w = w.copy() # Weight = [W1, W2] {nd.array}
        self.b = b # bias = {flaot}
        self.dw = None #derivative of weight
        self.db = None # derivative of bias
        self.lr = lr # learning rate
    
    def forward(self, x):
        self.x = x.copy()
        out = np.dot(self.x, self.w) + self.b
        return out
    
    def backward(self, dout):
        #get dW abd db
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        #update weight and bias
        self.w -= self.lr * self.dw
        self.b -= self.lr * self.db


class Single_layer_neural_network:
    def __init__(self, Weight, Bias):
        self.affine = Affine(Weight, Bias)
        self.sigmoid = Sigmoid()
        self.cee = cross_entropy()
        self.pre_error = float(9999)
        
    def Train(self, X_train, Y_train, epoch = 1000, batch_size = 1000):
        self.log = []
        self.X_train = X_train.copy()
        self.Y_train = Y_train.copy()
        self.batch_size = batch_size
        iter = len(X_train) / batch_size
        for _ in range(epoch):
            for i in range(int(iter + 1)):
                start = batch_size * i 
                end = batch_size * (i+1)
                if end > 1000:
                    end = 1000
                #One iter = train every sample form X_train once
                affine_out = self.affine.forward(self.X_train[start : end]) # W = [W1, W2], B = [B] / X[i] = {np.array}(2,),  -> W * X[i] + B = {float} 
                sig_out = self.sigmoid.forward(affine_out) # affine_out = {float} -> sig_out = 0 < {float} < 1
                cee_out = self.cee.forawrd(self.Y_train[start : end], sig_out) # error = - [log(y[p(y)]) + log([1-y][1-p(y)])]
                self.log.append(cee_out)
                if self.pre_error - cee_out < 0.0000000000000001:
                    print(f"pre = {self.pre_error}")
                    print(f"epoch = {_}, iteration = {i}, early stop")
                    return (self.affine.w, self.affine.b)

                self.pre_error = cee_out
                d_error = self.cee.backward() # backporpagation for crossentropy
                d_sig = self.sigmoid.backward(d_error)# backporpagation for sigmoid
                self.affine.backward(d_sig)#backporpagation for affine -> update weight and bias
                
        return (self.affine.w, self.affine.b)

    def predict(self, x):
        pred = []
        prob = np.dot(x, self.affine.w) + self.affine.b
        if prob > 0.5:
            pred.append(1)
        else:
            pred.append(0)
        return pred

def Accuracy(prediction, actual):
    score = 0
    for pred, label in zip(prediction, actual):
        if pred == label:
            score += 1
    return score / len(actual)
    