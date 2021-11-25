import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skl
import sklearn.neighbors as sk
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from sklearn.mixture import GaussianMixture as gm

#reading the csv file
df = pd.read_csv('SteelPlateFaults-2class.csv')
print(df)
#Name - Akshar Singh
#Roll no. - B20147
#Mobile no. - 7428357700
#Q1
# making test and train data 
d0 = df.groupby('Class').get_group(0)
d1 = df.groupby('Class').get_group(1)
d0_train,d0_test,d0_label_train,d0_label_test = skl.train_test_split(d0.iloc[:,0:27],d0['Class'],test_size=0.3,random_state=42,shuffle=True)
d1_train,d1_test,d1_label_train,d1_label_test = skl.train_test_split(d1.iloc[:,0:27],d1['Class'],test_size=0.3,random_state=42,shuffle=True)
train_data = pd.read_csv('SteelPlateFaults-train.csv')
test_data = pd.read_csv('SteelPlateFaults-test.csv')
d0_train=d0_train.drop(['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],axis=1)
d1_train=d1_train.drop(['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],axis =1)
test_data = test_data.drop(['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],axis =1)
final_label_test = pd.concat([d0_label_test,d1_label_test])

#predictions
lst = [2,4,8,16]
for i in lst:
    gmm0 = gm(n_components = i, covariance_type = 'full',reg_covar=1e-5)
    gmm0.fit(d0_train)
    gmm1 = gm(n_components = i, covariance_type = 'full',reg_covar = 1e-5)
    gmm1.fit(d1_train)
    pred = []
    prob0 = gmm0.score_samples(test_data)
    prob1 = gmm1.score_samples(test_data)
    for j in range(len(prob0)):
        if(prob0[j]>prob1[j]):
            pred.append(0)
        else:
            pred.append(1)
    print(f'confusion matrix for q = {i} is :-')
    print(confusion_matrix(final_label_test,pred))
    print(f'Accuracy score for q ={i} is ',accuracy_score(final_label_test,pred))




