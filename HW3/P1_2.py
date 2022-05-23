import pandas as pd
import numpy as np
import collections
import math 
import random


#Tranform str type class to int type
def Str_to_ID(label):
    pos_to_id = {}
    i = 0
    #give ID to every POS
    for pos in set(label):
        pos_to_id[pos] = i
        i += 1
    
    pos_id = []
    #create int type of label to replace 
    for pos in label:
        pos_id.append(pos_to_id[pos])
    
    return pos_id


#separate group by class to calculate mean and var
def separate_by_class(data):
    separated = {}
    for i in range(len(data)):
        vector = data[i]
        class_value = vector[-1]
        #It it is new class, create new dictionary key
        if (class_value not in separated):
            separated[class_value] = list()
        #append data into appropriate group
        separated[class_value].append(vector)
    return separated

#calculate mean and var by class
def calc_mean_var(sep):
    group_mean = {}
    group_var = {}

    for group in sep:
        mean_lst = []
        var_lst = []
        group_by_team = pd.DataFrame(sep[group])
        
        #execpt class columns
        for i in range(group_by_team.shape[1] - 1):
            #calculate mean and var of every attributes
            mean_lst.append(group_by_team.iloc[:,i].mean())
            var_lst.append(group_by_team.iloc[:,i].var())
        #and save with groupID as key
        group_mean[group] = mean_lst
        group_var[group] = var_lst
        
    mean_mat = []
    #trasform into matrix form
    for i in range(len(sep)):
        mean_mat.append(group_mean[i])

    var_mat = []
    for i in range(len(sep)):
        var_mat.append(group_var[i])        
    return np.array(mean_mat), np.array(var_mat)


#operating function for calculate mean and variance for multiple class
def MultiClass_mean_var(df):
    sep = separate_by_class(np.array(df))
    mean, var = calc_mean_var(sep)
    return mean, var


#calculate pre_probablity by multiple class
def MultiClass_pre_probablity(y):
    y_count = collections.Counter(y)
    pre_prob = np.ones(5)
    for i in range(0, 5):
        pre_prob[i] = y_count[i] / y.shape[0]
    return pre_prob

#calculate post_probality by x's attributs
def MultiClass_prob_feature_class(m, v, x):
    n_features = m.shape[1]
    pfc = np.ones(5)
    for i in range(0, 5):
        product = 1
        for j in range(0, n_features):
            product = product * (1/math.sqrt(2*3.14*v[i][j])) * math.exp(-0.5 * pow((x[j] - m[i][j]),2)/v[i][j])
        pfc[i] = product
    return pfc

#calculate accuracy of multi-class model
def multi_accu(Y_test, prediction):
    Y_test = np.array(Y_test)
    correct = 0
    for i in range(len(Y_test)):
        if Y_test[i] == prediction[i]:
            correct += 1
    return correct / len(Y_test)


#Naive bayes function for multiple class
def MultiClass_myNB(X_train, Y_train, X_test, Y_test):
    df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(Y_train)], axis = 1)

    mean, var = MultiClass_mean_var(df)

    prediction = []
    posterior_list = []
    pre_probab = MultiClass_pre_probablity(Y_train)
    
    for x_i in X_test.values:
        posterior_list.append(MultiClass_prob_feature_class(mean, var, x_i))
    for idx, posterior in enumerate(posterior_list):
        total_prob = 0
        for i in range(0, 5):
            total_prob = total_prob + (posterior[i] * pre_probab[i])
        for i in range(0, 5):
            posterior_list[idx][i] = (posterior[i] * pre_probab[i]) / total_prob
        pred = int(posterior_list[idx].argmax())
        prediction.append(pred)
    performance = multi_accu(Y_test, prediction)
    
    return prediction, posterior_list, performance




#fold
def cross_validation_split(dataset, n_folds):
    dataset = list(dataset.values)
    dataset_split = []
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(len(dataset))
            fold.append(dataset.pop(index))
        dataset_split.append(fold)
    return dataset_split
