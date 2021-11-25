
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sm
from statsmodels.tsa.ar_model import AutoReg as AR
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy as sp
from scipy import spatial as spatial
from sklearn.cluster import DBSCAN
#Name - Akshar Singh
#Roll no-B20147
#Mobile no. - 7428357700
#Reading the dataset
df = pd.read_csv('Iris.csv')
print(df)
red_df = df.iloc[:,:4]
print(red_df)

test = []
for i in df['Species']:
    if i == 'Iris-setosa':
        test.append(1)
    elif i == 'Iris-virginica':
        test.append(2)
    else:
        test.append(0)

#1
#calculating eighen values and eighen vectors
cov_mat = np.cov(red_df.T)
eig_vals,eig_vecs = np.linalg.eig(cov_mat) 
x=['eighen_val1','eighen_val2','eighen_val3','eighen_val4']
print(eig_vals)
plt.bar(x,eig_vals)
plt.show()
#Performing pca analysis
pca = PCA(n_components=2)
reduced_data = pd.DataFrame(pca.fit_transform(red_df),columns=['A','B'])
#Plotting the data
plt.scatter(reduced_data['A'],reduced_data['B'])
plt.show()

#2
#Kmeans clustering
K = 3
kmeans = KMeans(n_clusters=K)
kmeans.fit(reduced_data)
kmeans_prediction = kmeans.predict(reduced_data)
centres = kmeans.cluster_centers_
#a
#Scatter-plot
plt.scatter(reduced_data['A'],reduced_data['B'], c=kmeans_prediction)
plt.scatter([centres[i][0] for i in range(K)],[centres[i][1] for i in range(K)],marker='s',color='black',label='cluster_centre')
plt.legend()
plt.title('KNN for k=3')
plt.xlabel('A')
plt.ylabel('B')
plt.plot()
plt.show()
#b
print('The distortion measure for k =3 is',kmeans.inertia_)
#c
def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 #print(contingency_matrix)
 # Find optimal one-to-one mapping between cluster labels and true labels
 row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
 return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
print('The purity score for k =3 is',purity_score(test, kmeans_prediction))

#3
#using Kmeans clustering
K_vals = [2,3,4,5,6,7]
distortion = []
for K in K_vals:
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(reduced_data)
    kmeans_prediction = kmeans.predict(reduced_data)
    print(f'The distortion measure for k ={K} is',kmeans.inertia_)
    print(f'The purity score for k ={K} is',purity_score(test, kmeans_prediction))
    distortion.append(kmeans.inertia_)
#Plot of K VS DISTORTION measure
plt.plot(K_vals,distortion)
plt.title("KMeans")
plt.xlabel('K values')
plt.ylabel("Distortion measure")
plt.show()

#4
# GMM
from sklearn.mixture import GaussianMixture
K = 3
gmm = GaussianMixture(n_components = K)
gmm.fit(reduced_data)
GMM_prediction = gmm.predict(reduced_data)
centres = gmm.means_
#a
#Scatter-plot
plt.scatter(reduced_data['A'],reduced_data['B'], c=GMM_prediction)
plt.scatter([centres[i][0] for i in range(K)],[centres[i][1] for i in range(K)],marker='s',color='black',label='cluster_centre')
plt.legend()
plt.title('GMM for k=3')
plt.xlabel('A')
plt.ylabel('B')
plt.plot()
plt.show()

#b
#total log likelihood
print('The distortion measure(GMM) for k =3 is', gmm.lower_bound_*len(reduced_data))
#c
print('The purity score for(GMM) k =3 is',purity_score(test, GMM_prediction))
#5
K_vals = [2,3,4,5,6,7]
distortion = []
for K in K_vals:
    gmm = GaussianMixture(n_components = K)
    gmm.fit(reduced_data)
    GMM_prediction = gmm.predict(reduced_data)
    print(f'The distortion measure(GMM) for k ={K} is',gmm.lower_bound_*len(reduced_data))
    print(f'The purity score(GMM) for k ={K} is',purity_score(test, GMM_prediction))
    distortion.append(gmm.lower_bound_*len(reduced_data))
#Plot of k VS DISTORTION measure
plt.plot(K_vals,distortion)
plt.title("GMM")
plt.xlabel('K values')
plt.ylabel("Distortion measure")
plt.show()

#6
#DBSCAN
eps = [1,1,5,5]
min_samples = [4,10,4,10]
for i in range(len(eps)):
 dbscan_model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(reduced_data)
 DBSCAN_predictions = dbscan_model.labels_
 #Scatter plot and purity score
 plt.scatter(reduced_data['A'], reduced_data['B'], c=DBSCAN_predictions)
 print(f'The purity for eps = {eps[i]} and min_samples = {min_samples[i]} is ',purity_score(test,DBSCAN_predictions))
 plt.title(f"DBSCAN for eps = {eps[i]} and min_samples={min_samples[i]}")
 plt.show()