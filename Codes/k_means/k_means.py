import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

iris = datasets.load_iris()
x = pd.DataFrame(iris.data,columns=['Sepal_length','Sepal_width','Petal_length','Petal_width'])
y = pd.DataFrame(iris.target,columns=['Target'])

colormap = np.array(['red','lime','black'])
plt.figure(figsize=(14,7))
model = KMeans(n_clusters=3)
model.fit(x)
model.labels_

plt.subplot(2,1,2)
plt.scatter(x.Sepal_length,x.Sepal_width,c=colormap[y.Target],s=40)
plt.title('Real Classification')

plt.subplot(1,2,1)
plt.scatter(x.Sepal_length,x.Sepal_width,c=colormap[model.labels_],s=40)
plt.title('calc Classification')

print('classification_report',classification_report(y.Target,model.labels_))

distortion = []
K = range(1,10)
for k in K:
    model = KMeans(n_clusters=k)
    model.fit(x)
    distortion.append(model.inertia_)

print(distortion)

plt.figure(figsize=(16,7))
plt.plot(K,distortion,'-bx')
plt.show()
