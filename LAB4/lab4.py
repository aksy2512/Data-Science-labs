import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skl
import sklearn.neighbors as sk
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
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

a=[d0_train,d1_train]
final_train = pd.concat(a)
final_test = pd.concat([d0_test,d1_test])
final_label_train = pd.concat([d0_label_train,d1_label_train])
final_label_test = pd.concat([d0_label_test,d1_label_test])
final_test.to_csv('SteelPlateFaults-test.csv',index=False)
final_train.to_csv('SteelPlateFaults-train.csv',index=False)

# Using the Knn classifier and calculating confusion matrix and accuracy score
for i in range(1,6,2):
 classifier = sk.KNeighborsClassifier(n_neighbors=i)
 classifier.fit(final_train,final_label_train)

 pred = classifier.predict(final_test)
 print(f'For k = {i} :')
 print(confusion_matrix(final_label_test,pred))
 print(accuracy_score(final_label_test,pred))
 
#Q2
#Data preprocessing - Min max normalisating
final_train_scaled = final_train.copy()
final_test_scaled = final_test.copy()
for column in final_train.columns:
   final_train_scaled[column] = (final_train[column] - final_train[column].min()) / (final_train[column].max() - final_train[column].min())
   final_test_scaled[column] = (final_test[column] - final_train[column].min()) / (final_train[column].max() - final_train[column].min())
print(final_test_scaled)
print(final_train_scaled)
final_train_scaled.to_csv('SteelPlateFaults-train-Normalised.csv',mode='w',index=False)
final_test_scaled.to_csv('SteelPlateFaults-test-normalised.csv',mode = 'w',index=False)

# Using the Knn classifier and calculating confusion matrix and accuracy score
for i in range(1,6,2):
 classifier = sk.KNeighborsClassifier(n_neighbors=i)
 classifier.fit(final_train_scaled,final_label_train)

 pred = classifier.predict(final_test_scaled)
 print(f'For k = {i} :')
 print(confusion_matrix(final_label_test,pred))
 print(accuracy_score(final_label_test,pred))

#3
#Preparing test and train data
train_data = pd.read_csv('SteelPlateFaults-train.csv')
test_data = pd.read_csv('SteelPlateFaults-test.csv')
d0_train=d0_train.drop(['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],axis=1)
d1_train=d1_train.drop(['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],axis =1)
test_data = test_data.drop(['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],axis =1)
#Computing mean and covariance matrix
mean_vec0 = d0_train.mean()
cov0=d0_train.cov()
cov0 = pd.DataFrame(cov0)
cov0.to_csv('Cov0.csv')
mean_vec1 = d1_train.mean()
print(mean_vec1.to_list())
cov1=d1_train.cov()
p0 = len(d0.index)/len(df.index)
p1 = len(d1.index)/len(df.index)
cov1.to_csv('Cov1.csv')
#Computing the probabilities for each class
pred = []
print(test_data-mean_vec0)
l0 = np.exp((-0.5)*np.dot(np.dot(test_data-mean_vec0,np.linalg.inv(cov0)),(test_data-mean_vec0).T).diagonal())/(pow(2*np.pi,25/2)*pow(abs(np.linalg.det(cov0)),0.5))
l1 = np.exp((-0.5)*np.dot(np.dot(test_data-mean_vec1,np.linalg.inv(cov1)),(test_data-mean_vec1).T).diagonal())/(pow(2*np.pi,25/2)*pow(abs(np.linalg.det(cov1)),0.5))

tot = l0*p0 + l1*p1
t0 = l0*p0/tot
t1 = l1*p1/tot
#Comparing probabilities and computing confusion matrix and accuracy score.
for i in range(len(t0)):
    if(t0[i]>t1[i]):
        pred.append(0)
    else:
        pred.append(1)
print(confusion_matrix(final_label_test,pred))
print(accuracy_score(final_label_test,pred))
    
  

  



