import numpy as np
import P1_1
from tqdm import tqdm
import matplotlib.pyplot as plt


#Sigmoid
class Sigmoid:
    def __init__(self):
        self.y = None
    
    #forward fomular for sigmoid
    def forward(self, X):
        self.y = 1 / (1+np.exp(-X))
        return self.y
    
    #get dout and multiply with backward of sigmoid
    def backward(self, dout):
        dout_lst = dout * self.y * (1.0 - self.y)
        return dout_lst
    
class cross_entropy:
    def __init__(self):
        self.cee_out = None
    
    #forward
    def forawrd(self, y, sig_out):
        self.y = y.copy()
        self.sig_out = sig_out.copy()
        #if 0 or 1 go inside of log, It return 0 or -Inf
        #To avoid that situatuion, transform 0 into very low value and 1 into very close to 1
        temp = np.where(self.sig_out == 0, 0.000000001, self.sig_out)
        temp = np.where(temp == 1, 0.999999999, temp)
        self.cee_out = -(  (self.y * np.log(temp) ) + ( (1 - self.y) * np.log(1 - temp) ) )
        return np.mean(self.cee_out)
    
        
    def backward(self):
        #It already got self.y and self.sigout value. don't need any input
        return (self.sig_out - self.y) / self.sig_out * (1 - self.sig_out)
    
    
#Affine class
class Affine:
    def __init__(self, w, b, lr):
        self.x = None # X = [x1, x2] {nd.array}
        self.w = w.copy() # Weight = [W1, W2] {nd.array}
        self.b = b # bias = {flaot}
        self.dw = None #derivative of weight
        self.db = None # derivative of bias
        self.lr = lr # learning rate
    
    def forward(self, x):
        self.x = x.copy()
        # w1x1 + w2x2 + b
        out = np.dot(self.x, self.w) + self.b
        return out
    
    def backward(self, dout):
        #get dW abd db
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        grad_norm = np.linalg.norm([self.dw[0], self.dw[1], self.db], ord = 2)
        #update weight and bias
        self.w -= self.lr * self.dw
        self.b -= self.lr * self.db
        return grad_norm


#class for ANN
class Single_layer_neural_network:
    def __init__(self, Weight, Bias, lr = 1):
        self.affine = Affine(Weight, Bias, lr)
        self.sigmoid = Sigmoid()
        self.cee = cross_entropy()
        
    def Train(self, X_train, Y_train, epoch = 1000, batch_size = 1000):
        self.log = []
        self.X_train = X_train.copy()
        self.Y_train = Y_train.copy()
        self.batch_size = batch_size
        prev_err = 9999
        iter = len(X_train) / batch_size
        for curr_epoch in range(epoch):
            for i in range(int(iter + 1)):
                start = batch_size * i 
                end = batch_size * (i+1)
                #if end goes to out of index, make in 1000
                if end > 1000:
                    end = 1000
                    
                #avoid empth list    
                if start == end:
                    break
                #One iter = train every sample form X_train once
                affine_out = self.affine.forward(self.X_train[start : end]) # W = [W1, W2], B = [B] / X[i] = {np.array}(2,),  -> W * X[i] + B = {float} 
                sig_out = self.sigmoid.forward(affine_out) # affine_out = {float} -> sig_out = 0 < {float} < 1
                err = self.cee.forawrd(self.Y_train[start : end], sig_out)
                
                #early stop
                if (prev_err > err) and (prev_err - err < 0.00001):
                    print(f"Early Stop at epoch : {curr_epoch}, iteration : {i}")
                    return (self.affine.w, self.affine.b)
                prev_err = err
                self.log.append([err, self.affine.w.copy()])# error = - [log(y[p(y)]) + log([1-y][1-p(y)])]
                d_error = self.cee.backward() # backporpagation for crossentropy
                d_sig = self.sigmoid.backward(d_error)# backporpagation for sigmoid
                grad_norm = self.affine.backward(d_sig)#backporpagation for affine -> update weight and bias
                #ealry stop
                if grad_norm < 0.00000001:
                    print(f"Early Stop at epoch : {curr_epoch}, iteration : {i}, norm of gradient : {grad_norm}")
                    return (self.affine.w, self.affine.b)
        return (self.affine.w, self.affine.b)

    #Get probability
    def predict_proba(self, x):
        prob = np.dot(x, self.affine.w) + self.affine.b
        prob = 1 / (1+np.exp(-prob))
        return prob


#Function for accuracy
def Accuracy(prediction, actual):
    score = 0
    for pred, label in zip(prediction, actual):
        if pred == label:
            score += 1
    return score / len(actual)


#Function for plot the ROC curve
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()