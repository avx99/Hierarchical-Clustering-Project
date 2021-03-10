import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('default of credit card clients.csv')
# df.replace(" ",np.nan,inplace=True)
df =df[:500]
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

# from sklearn import preprocessing
# labelencoder = preprocessing.LabelEncoder()
# y = labelencoder.fit_transform(y)

# a = pd.get_dummies(df['sex'])
# a = a.iloc[:,1]
# df['Sex'] = a

# a = pd.get_dummies(df['sex'])
# df.drop(['sex'],axis=1,inplace=True)
# df['sex'] = a.iloc[:,0]

# a = pd.get_dummies(df['education'])
# df.drop(['education'],axis=1,inplace=True)
# df['education 1'] = a.iloc[:,0]
# df['education 2'] = a.iloc[:,1]
# df['education 3'] = a.iloc[:,2]
# df['education 4'] = a.iloc[:,3]
# df['education 5'] = a.iloc[:,4]

# # a = pd.get_dummies(df['product_type'])
# df.drop(['product_type'],axis=1,inplace=True)
# # df['Checking account 1'] = a.iloc[:,0]
# # df['Checking account 2'] = a.iloc[:,1]

# a = pd.get_dummies(df['family_status'])
# df.drop(['family_status'],axis=1,inplace=True)
# df['family_status 1'] = a.iloc[:,0]
# df['family_status 2'] = a.iloc[:,1]
# # df['Purpose 2'] = a.iloc[:,1]
# # df['Purpose 3'] = a.iloc[:,0]
# # df['Purpose 4'] = a.iloc[:,1]
# # df['Purpose 5'] = a.iloc[:,0]
# # df['Purpose 6'] = a.iloc[:,1]
# # df['Purpose 7'] = a.iloc[:,0]




from sklearn.decomposition import PCA
acp = PCA()

from sklearn import preprocessing
Z = preprocessing.StandardScaler().fit_transform(X)

moyennes = np.mean(Z,axis=0)
var = np.std(Z,axis=0)

composants = acp.fit_transform(Z)
lambdas = np.std(composants,axis=0)


n = 100
fig, ax = plt.subplots(figsize=(15,15))
ax.plot(composants[:,0],composants[:,1],"wo")
ax.axis([-4,+4,-4,+4])
ax.plot([-4,+4],[0,0],color='silver',linestyle='--')
ax.plot([0,0],[-4,+4],color='silver',linestyle='--')
ax.set_xlabel("Comp.1 "+ str(100*lambdas[0]/sum(lambdas))+" %")
ax.set_ylabel("Comp.2 "+ str(100*lambdas[1]/sum(lambdas))+" %")
for i in range(n):
    ax.text(composants[i,0],composants[i,1],i)
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
dendogram = dendrogram(linkage(Z[:20],method = 'ward'))
plt.title('dendogramme')
plt.xlabel('les individus')
plt.ylabel('la distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
target = hc.fit_predict(X)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, target)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
cm_kmeans = confusion_matrix(y, y_kmeans)



plt.scatter(composants[target == 0, 0], composants[target == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(composants[target == 1, 0], composants[target == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlim((-10,30))
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()




