import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.graphics.tsaplots as sm
from statsmodels.tsa.ar_model import AutoReg as AR
import math
#Reading the data
df = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
print(df)
#Name - Akshar Singh
#Roll no - B20147
#Mobile no - 7428357700
#Q1) 
#a
#Line plot of day with number of cases
plt.plot([i for i in range(1,613,1)],df['new_cases'])
plt.xlabel('Days')
plt.ylabel('Confirmed cases')
plt.show()

#b
#one lag sequence 
df_one = df.iloc[:611,:]
#calculating correlation value
corr = np.corrcoef(df_one['new_cases'],df.iloc[1:612,:]['new_cases'])
print(corr[0][1])

#c
#plotting scatter plot
plt.scatter(df.iloc[1:612,:]['new_cases'],df_one['new_cases'])
plt.xlabel('No time lag')
plt.ylabel('One-day time lag')
plt.show()

#d
days = [1,2,3,6]
corrval = []
#computing correlation values for 1,2,3,6 time lag
for i in days:
    corrval.append(np.corrcoef(df.iloc[:612-i,:]['new_cases'],df.iloc[i:612,:]['new_cases'])[0][1])
    print(f'Correlation value for {i} day lag is ',np.corrcoef(df.iloc[:612-i,:]['new_cases'],df.iloc[i:612,:]['new_cases'])[0][1] )
#plotting the line plot
plt.plot(days,corrval)
plt.xlabel('Time lag')
plt.ylabel('Correlation coefficient values')
plt.show()

#e
#Plotting a correlogram
sm.plot_acf(df['new_cases'],lags =10)
plt.show()

#Q2
#splitting the test-train data
test_size = 0.35
X = df.values
tst_sz = math.ceil(len(X)*test_size)
train,test = X[:len(X)-tst_sz],X[len(X)-tst_sz:]
#a
#plotting test,train data
plt.plot(test,color = 'r',label ='test')
plt.plot(train, color = 'b',label = 'train')
plt.xlabel('Day')
plt.ylabel('Covid cases')
plt.legend()
plt.show()

#Modeling our data by Autoregression
window = 5
model = AR(train,lags=window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 
print(coef)
#b
#prediction
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window :]

history = [history[i] for i in range(len(history))]
print(history)
predictions = [] # List to hold the predictions, 1 step at a time
for t in range(len(test)):
 length = len(history)
 lag = [history[i] for i in range(length-window,length)]
 yhat = coef[0] # Initialize to w0
 for d in range(window):
  yhat += coef[d+1] * lag[window-d-1] # Add other values
 predictions.append(yhat) #Append predictions to compute RMSE
 obs = test[t]
  

 history.append(obs) # Append actual test value to history, to be used in the next step
#b i)
#Scatter plot
print(len(test))
print(len(predictions))
plt.scatter(test,predictions)
plt.xlabel('Actual data')
plt.ylabel('Predicted data')
plt.show()

#ii)
#line plot
plt.plot(test,label='test')
plt.plot(predictions,label='prediction')
plt.legend()
plt.show()

#iii)
#RMSE % calculation
rmse =0
tot =0
for i in range(len(test)):
    rmse = rmse + (predictions[i]-test[i])**2
    tot += test[i]
rmse = ((rmse/len(test))**0.5)*100/(tot/len(test))
print(f'The RMSE(%) error is {rmse}')
mape = np.mean(np.abs((test-predictions)/test))*100
print(f'The mape  error is {mape}')

#c)
rmse_list = []
mape_list = []
val = [1,5,10,15,25]
for window in val:
    print(window)
    model = AR(train,lags=window)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model
    history = train[len(train)-window :]


    history = [history[i] for i in range(len(history))]
    
    predictions = [] # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
     length = len(history)
     lag = [history[i] for i in range(length-window,length)]
     yhat = coef[0] # Initialize to w0
     for d in range(window):
      yhat += coef[d+1] * lag[window-d-1] # Add other values
     predictions.append(yhat) #Append predictions to compute RMSE
     obs = test[t]
     history.append(obs)
    rmse =0
    tot =0
    for i in range(len(test)):
     rmse = rmse + (predictions[i]-test[i])**2
     tot += test[i]
    rmse = ((rmse/len(test))**0.5)*100/(tot/len(test))
    print(f'The RMSE(%) error for lag = {window} is {rmse}')
    mape = np.mean(np.abs((test-predictions)/test))*100
    print(f'The mape  error for lag = {window} is {mape}')
    rmse_list.append(rmse[0])
    mape_list.append(mape)
plt.bar(val,rmse_list)
plt.xlabel('Time lag values')
plt.ylabel('RMSE(%) values')
plt.show()
plt.xlabel('Time lag values')
plt.ylabel('MAPE(%) values')
plt.bar(val,mape_list)
plt.show()

#d)
#Calculating the heurestic value
heurestic =0
for i in range(1,100,1):
   corr = np.corrcoef(df.iloc[:len(train)-i,:]['new_cases'],df.iloc[i:len(train),:]['new_cases'])[0][1]
   print(corr)

   if(abs(corr)<2/(len(train)**0.5)):
     heurestic=i-1
     break
print(f'The heurestic value for time lag is {heurestic}')
window = heurestic
model = AR(train,lags=window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 

#b
#prediction
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window :]


history = [history[i] for i in range(len(history))]

predictions = [] # List to hold the predictions, 1 step at a time
for t in range(len(test)):
 length = len(history)
 lag = [history[i] for i in range(length-window,length)]
 yhat = coef[0] # Initialize to w0
 for d in range(window):
  yhat += coef[d+1] * lag[window-d-1] # Add other values
 predictions.append(yhat) #Append predictions to compute RMSE
 obs = test[t]
  

 history.append(obs) # Append actual test value to history, to be used in the next step
#RMSE % calculation
rmse =0
tot =0
for i in range(len(test)):
    rmse = rmse + (predictions[i]-test[i])**2
    tot += test[i]
rmse = ((rmse/len(test))**0.5)*100/(tot/len(test))
print(f'The RMSE(%) error is {rmse}')
mape = np.mean(np.abs((test-predictions)/test))*100
print(f'The mape  error is {mape}')

#Extra work
test_size = 0
X = df.values
tst_sz = math.ceil(len(X)*test_size)
train,test = X[:len(X)-tst_sz],X[len(X)-tst_sz:]
print(train)
window = heurestic
model = AR(train,lags=window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 


#prediction
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window :]


history = [history[i] for i in range(len(history))]

predictions = [] # List to hold the predictions, 1 step at a time
for t in range(1,121,1):
 length = len(history)
 lag = [history[i] for i in range(length-window,length)]
 yhat = coef[0] # Initialize to w0
 for d in range(window):
  yhat += coef[d+1] * lag[window-d-1] # Add other values
 predictions.append(yhat) #Append predictions to compute RMSE
 
  

 history.append(yhat) # Append actual test value to history, to be used in the next step
plt.plot(predictions)
plt.show()



