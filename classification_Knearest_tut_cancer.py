"""This has 2 types of classification -->KNN and SVM"""
    
import pandas as pd
import numpy as np
from sklearn import preprocessing,neighbors,cross_validation,svm
    
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
    
X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])
    
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)
    
#clf = neighbors.KNeighborsClassifier()
clf = svm.SVC()
    
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(accuracy,prediction)
    
