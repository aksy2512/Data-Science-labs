import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics

#os.chdir("../Q1")
df = pd.read_csv("Q1/pima-indians-diabetes.csv")
#Q1
print(df.describe())
print(df.mode())

#Q2
#a
rows=['pregs','plas','pres','skin','test','BMI','pedi']
fig,axs = plt.subplots(3,3,figsize=(8,4))
k=0
for i in range(0,2):
    for j in range(0,3):
        axs[i][j].scatter(df['Age'],df[rows[k]],s=10,marker='o')
        axs[i][j].set_title(f'Plot between age and {rows[k]}')
        
        k=k+1
axs[2][0].scatter(df['Age'],df[rows[k]],s=10,marker='o')
axs[2][0].set_title(f'Plot between age and {rows[k]}')
c=0        
for ax in axs.flat:
    ax.set(xlabel='',ylabel=rows[c])
    
    c=c+1
    if(c==7):
        break
plt.suptitle("Q2 a")
plt.show()    
#b
rows=['pregs','plas','pres','skin','test','Age','pedi']
fig,axs = plt.subplots(3,3,figsize=(12,4))
k=0
for i in range(0,2):
    for j in range(0,3):
        axs[i][j].scatter(df['BMI'],df[rows[k]],s=10)
        axs[i][j].set_title(f'Plot between BMI and {rows[k]}')
        
        k=k+1
c=0
axs[2][0].scatter(df['BMI'],df[rows[k]],s=10,marker='o') 
axs[2][0].set_title(f'Plot between BMI and {rows[k]}')       
for ax in axs.flat:
    ax.set(xlabel='BMI',ylabel=rows[c])
    c=c+1
    if(c==7):
        break
plt.suptitle("Q2 b")
plt.show() 

#Q3
#a
rows=['pregs','plas','pres','skin','test','BMI','pedi']
for i in rows:
    corr = np.corrcoef(df['Age'],df[i])[0,1]
    print(f'Correlation coefficient of Age with {i} is {corr}')
print("----------------------------------------------------------------")    
#b
rows=['pregs','plas','pres','skin','test','Age','pedi']
for i in rows:
    corr = np.corrcoef(df['BMI'],df[i])[0,1]
    print(f'Correlation coefficient of BMI with {i} is {corr}')
print("---------------------------------------------------------------")    

#Q4
fig,axs = plt.subplots(2)
axs[0].hist(df['pregs'],bins=np.arange(0,20,1),edgecolor='black')
axs[0].set_xticks(range(21))
axs[0].set(ylabel='Frequencies',xlabel='pregs')

axs[1].hist(df['skin'],bins=np.arange(0,101,10),edgecolor='black')
axs[1].set_xticks(np.arange(0,101,10))
axs[1].set(ylabel='Frequencies',xlabel='skin in mm')
plt.suptitle('Q4')
    
plt.show()

#Q5
new0 = df.loc[df['class']==0]
new1 = df.loc[df['class']==1]
fig,axs = plt.subplots(2)
axs[0].hist(new0['pregs'],bins=np.arange(0,20,1),edgecolor='black')
axs[0].set_xticks(range(21))
axs[0].set(ylabel='Frequencies',xlabel='pregs for class0')

axs[1].hist(new1['pregs'],bins=np.arange(0,21,1),edgecolor='black')
axs[1].set_xticks(np.arange(0,21,1))
axs[1].set(ylabel='Frequencies',xlabel='pregs for class1')
plt.suptitle('Q5')
    
plt.show()

#6
rows=['pregs','plas','pres','skin','test','BMI','pedi','Age']
fig,axs = plt.subplots(2,4)
k=0
for i in range(2):
    for j in range(4):
     axs[i][j].boxplot(df[rows[k]])
     axs[i][j].set_title(rows[k])
     k=k+1
for i in rows:
    print(f'Variance of {i} is: ', statistics.variance(df[i]))
plt.show()