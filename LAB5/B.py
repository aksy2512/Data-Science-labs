import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skl
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
import multipolyfit as mpf
df = pd.read_csv('abalone.csv')
#test train data
in_train, in_test, out_train, out_test = skl.train_test_split(df.iloc[:,0:7], df['Rings'], test_size = 0.3,random_state=42)
in_train.to_csv('abalone-train.csv',index=False)
in_test.to_csv('abalone-test.csv',index=False)



cols = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight']
#Q1
#setting input attribute
val_input = ''
ma=-1
for i in cols:
    if(pearsonr(out_train,in_train[i])[0]>ma):
        ma = pearsonr(out_train,in_train[i])[0]
        val_input = i

#Plotting the line of best fit
m,b = np.polyfit(in_train[val_input],out_train,1)
plt.plot(in_train[val_input],m*in_train[val_input]+b)
plt.scatter(in_train[val_input],out_train,color = 'orange')
plt.show()
#prediction using linear regression
X = np.array(in_train[val_input]).reshape(-1,1)
X_test = np.array(in_test[val_input]).reshape(-1,1)
reg = LinearRegression().fit(X,out_train)
#Root mean square error for train data
predinp = reg.predict(X)

rsme1 = np.sqrt(((predinp-out_train)**2).mean())
print('Root mean square error for train data is ',rsme1)
#Root mean square error for test data
predtest = reg.predict(X_test)

rsme2 = np.sqrt(((predtest-out_test)**2).mean())
print('Root mean square error for test data is ',rsme2)
#Plotting scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data
plt.scatter(out_test,predtest)
plt.show()

#Q2
reg2 = LinearRegression().fit(in_train,out_train)
predinp = reg2.predict(in_train)
#Root mean square error for train data
rsme1 = np.sqrt(((predinp-out_train)**2).mean())
print('Root mean square error for train data for Q2 is ',rsme1)
#Root mean square error for test data
predtest = reg2.predict(in_test)
rsme2 = np.sqrt(((predtest-out_test)**2).mean())
print('Root mean square error for test data for Q2 is ',rsme2)
#Plotting scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data
plt.scatter(out_test,predtest)
plt.show()

#Q3
rmse_train = []
rmse_test = []
deg = [2,3,4,5]
for i in range(2,6,1):
    #Polynomial expansion for input variable
    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(X)
    #Predicting using linear regressor
    regressor = LinearRegression()
    regressor.fit(x_poly,out_train)
    predinp = regressor.predict(x_poly)
    #Root mean square error for training data
    
    rsme1 = np.sqrt(((predinp-out_train)**2).mean())
    print(f'Root mean square error for train data for degree {i} is ',rsme1)
    rmse_train.append(rsme1)

    #Root mean square error for test data
    x_test_poly = poly_features.fit_transform(X_test)
    predtest = regressor.predict(x_test_poly)
    rsme2 = np.sqrt(((predtest-out_test)**2).mean())
    print(f'Root mean square error for test data for degree {i} is ',rsme2)
    rmse_test.append(rsme2)


#Bar plots:-
plt.bar([2,3,4,5],rmse_train)
plt.title('RMSE FOR TRAINING DATA')
plt.show()
plt.bar([2,3,4,5],rmse_test)
plt.title('RMSE FOR TEST DATA')
plt.show()

#Plotting the line of best fit
x=np.linspace(0,1,2923).reshape(-1,1)
mindeg = deg[rmse_train.index(min(rmse_train))]
poly_features = PolynomialFeatures(mindeg)
x_poly = poly_features.fit_transform(X)
poly = poly_features.fit_transform(x)
regressor = LinearRegression()
regressor.fit(x_poly,out_train)
x_test_poly = poly_features.fit_transform(X)
predtrain = regressor.predict(poly)
plt.plot(np.linspace(0,1,2923),predtrain,linewidth=3)
plt.title('Line of best fit for Q3')
plt.scatter(in_train[val_input],out_train,color = 'orange')
plt.show()
#finding the best fitting and scatter plot
mindeg = deg[rmse_test.index(min(rmse_test))]
poly_features = PolynomialFeatures(mindeg)
x_poly = poly_features.fit_transform(X)
regressor = LinearRegression()
regressor.fit(x_poly,out_train)
x_test_poly = poly_features.fit_transform(X_test)
predtest = regressor.predict(x_test_poly)
plt.scatter(out_test,predtest)
plt.title('Q3')
plt.show()
#Q4
rmse_train = []
rmse_test = []

for i in range(2,6,1):
    #Polynomial expansion for input variable
    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(in_train)
    #Predicting using linear regressor
    regressor = LinearRegression()
    regressor.fit(x_poly,out_train)
    predinp = regressor.predict(x_poly)
    #Root mean square error for training data
    
    rsme1 = np.sqrt(((predinp-out_train)**2).mean())
    rmse_train.append(rsme1)
    print(f'Root mean square error for train data for Q4 and degree {i} is ',rsme1)
    #Root mean square error for test data
    x_test_poly = poly_features.fit_transform(in_test)
    predtest = regressor.predict(x_test_poly)
    rsme2 = np.sqrt(((predtest-out_test)**2).mean())
    print(f'Root mean square error for test data for Q4 and degree {i} is ',rsme2)
    rmse_test.append(rsme2)
#Bar plots:-
plt.bar([2,3,4,5],rmse_train)
plt.title('RMSE FOR TRAINING DATA')
plt.show()
plt.bar([2,3,4,5],rmse_test)
plt.title('RMSE FOR TEST DATA')
plt.show()
#finding the best fitting and Scatterplot
mindeg = deg[rmse_test.index(min(rmse_test))]
poly_features = PolynomialFeatures(mindeg)
x_poly = poly_features.fit_transform(in_train)
regressor = LinearRegression()
regressor.fit(x_poly,out_train)
x_test_poly = poly_features.fit_transform(in_test)
predtest = regressor.predict(x_test_poly)
plt.scatter(out_test,predtest)
plt.show()











