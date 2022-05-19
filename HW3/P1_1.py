from random import randrange
import pandas as pd
import numpy as np
import math
import collections



#Generate sample data
def Generate_train_test_data(n_sample):
    #generate train data set
    train_data1 = np.random.multivariate_normal((1, 0), [[1, 0.75], [0.75, 1]], n_sample)
    train_data2 = np.random.multivariate_normal((0, 1.5), [[1, 0.75], [0.75, 1]], n_sample)
    train_df1 = pd.DataFrame(train_data1)
    train_df2 = pd.DataFrame(train_data2)
    train_df1['class'] = 0
    train_df2['class'] = 1
    
    train = pd.concat([train_df1, train_df2])
    train.reset_index(inplace = True, drop = True)
    train.columns = ["X1", "X2", "class"]
    train = train.sample(frac = 1).reset_index(drop = True)
    
    
    #generate test data set
    test_data1 = np.random.multivariate_normal((1, 0), [[1, 0.75], [0.75, 1]], 200)
    test_data2 = np.random.multivariate_normal((0, 1.5), [[1, 0.75], [0.75, 1]], 200)
    test_df1 = pd.DataFrame(test_data1)
    test_df2 = pd.DataFrame(test_data2)
    test_df1['class'] = 0
    test_df2['class'] = 1
    
    #transform into daframe
    test = pd.concat([test_df1, test_df2])
    test.reset_index(inplace = True, drop = True)
    test.columns = ["X1", "X2", "class"]
    test = test.sample(frac = 1).reset_index(drop = True)
    
    
    X_train = np.array(train.iloc[:, [0, 1]])
    Y_train = np.array(train['class'])

    X_test = np.array(test.iloc[:, [0, 1]])
    Y_test = np.array(test['class'])
    
    return X_train, Y_train, X_test, Y_test


#calculate prior probablity
def pre_probablity(y):
    #count each class in y
    y_count = collections.Counter(y)
    pre_prob = np.ones(2)
    for i in range(0, 2):
        pre_prob[i] = y_count[i] / y.shape[0]
    return pre_prob


#calculate mean and var
#########이 부분 함수가 직관적이지 않고 쓸모없이 길어서
#########교체 해야할 듯
####핵심은 class별로 각 attribute의 평균과 분산을 구해줌
###예를 들어서 class가 0인 데이터들의 X1의 평균은 1,  X2의 평균은 2이며
###class가 1인 데이터들의 X1의 평균은 2, X2의 평균은 3이라고 할때
###mean = [[1, 2], [2,3]] 으로 반환해주는 함수
def mean_var(x, y):
    num_of_feat = x.shape[1]
    #create empty array for store mean and var value
    mean = np.ones((2, num_of_feat))
    var = np.ones((2, num_of_feat))
    
    #count frqeuncy of each class
    n_0 = np.bincount(y)[np.nonzero(np.bincount(y))[0]][0]
    x0 = np.ones((n_0, num_of_feat))
    x1 = np.ones((x.shape[0] - n_0, num_of_feat))

    k = 0
    for i in range(0, x.shape[0]):
        if y[i] == 0:
            x0[k] = x[i]
            k += 1
            
    k = 0
    for i in range(0, x.shape[0]):
        if y[i] == 1:
            x1[k] = x[i]
            k += 1
            
    for j in range(0, num_of_feat):
        mean[0][j] = np.mean(x0.T[j])
        var[0][j] = np.var(x0.T[j]) * (n_0 / (n_0 - 1))
        mean[1][j] = np.mean(x1.T[j])
        var[1][j] = np.var(x1.T[j]) *  ( (x.shape[0] - n_0) / ((x.shape[0] - n_0) -1) )
    
    return mean, var

#calculate post probablity
def prob_feature_class(m, v, x):
    #define # of features by counting width of m
    n_features = m.shape[1]
    #define # of class (일단 여기서는 바이너리라고 가정하고 2를 그냥 줘버리는데, 이런 부분 고치면 multi NB로 하나의 함수로 통일 가능할듯)
    pfc = np.ones(2)
    
    #calculate pfc for every class
    for i in range(0, 2):
        product = 1
        #claculate pfc of each class by multiply every probablity from each attributes
        for j in range(0, n_features):
            product = product * (1/math.sqrt(2*3.14*v[i][j])) * math.exp(-0.5 * pow((x[j] - m[i][j]),2)/v[i][j])
        pfc[i] = product
    return pfc


#Create confusion matrix
def Create_Confusion_matrix(prediction, label):
    confusion_matrix = np.array([[0,0],
                                [0,0]])
    for i in range(len(prediction)):
        #If correct
        if prediction[i] == label[i]:
            #and It is TP
            if label[i] == 1:
                confusion_matrix[0][0] += 1
            #or It's FN
            else:
                confusion_matrix[1][1] += 1
        #If wrong 
        else:
            #It is TN
            if label[i] == 1:
                confusion_matrix[0][1] += 1
            #of FP
            else:
                confusion_matrix[1][0] += 1
    return confusion_matrix

#Calculate precision,recall, accuracy
def calc_accuracy(actual, pred):
    
    #First, create Confusion matrix, Than It is easy to calculate Precision, recall, accuracy
    conf_mat = Create_Confusion_matrix(actual, pred)
    precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
    recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    accuracy = (conf_mat[0][0] + conf_mat[1][1]) / (conf_mat[0][0] + conf_mat[1][1] + conf_mat[0][1] + conf_mat[1][0])
    
    #return by list
    return [precision, recall, accuracy]



#Operating function for binary Naive bayes
def myNB(X_train, Y_train, X_test, Y_test):
    
    #Get mean and var for calculate posterior
    mean, var = mean_var(X_train, Y_train)
    #create empty list for contain predictions and posteriors
    prediction = []
    posterior_list = []
    #calculate prior
    pre_probab = pre_probablity(Y_train)
    
    #calculate and save for each data in X_test
    for x_i in X_test:
        posterior_list.append(prob_feature_class(mean, var, x_i))

    for idx, posterior in enumerate(posterior_list):
        total_prob = 0
        #calculate total probality to use as denominator of posterior
        for i in range(0, 2):
            total_prob = total_prob + (posterior[i] * pre_probab[i])
        #calculate final posterior and save
        for i in range(0, 2):
            posterior_list[idx][i] = (posterior[i] * pre_probab[i])/total_prob
        #predict as class which has highest posterior probablity 
        pred = int(posterior_list[idx].argmax())
        prediction.append(pred)
    
    #calculate performance of this model (output = [precision, recall, accuracy])
    performance = calc_accuracy(Y_test, prediction)
    
    return prediction, posterior_list, performance


# Run Naive bayes for {iteration} times
def Run_myNB(iter = 1, n_sample = 500):
    #create empty list for contain result of every iterations
    pred_lst = []
    post_lst = []
    err_lst = []
    for i in range(iter):
        X_train, Y_train, X_test, Y_test = Generate_train_test_data(n_sample)
        
        prediction, posterior, err = myNB(X_train, Y_train, X_test, Y_test)
        
        pred_lst.append(prediction)
        post_lst.append(posterior)
        err_lst.append(err)
        
    return pred_lst, post_lst, err_lst



