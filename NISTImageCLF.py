import os
import cv2
import numpy as np
import _pickle as pickle
from time import time
from dataObj import dataObj
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def dataset():
    tp = time()
    xt = []
    yt = []
    for i in range(31):
        f = open('/home/adit/Desktop/ML/NIST/Dataset/dataset_'+(str)(i+1)+'.pkl','rb')
        m = pickle.load(f)
        print(len(m.x))
        xt = xt+m.x
        yt = yt+m.y
        f.close()
    print('Unpickling Time = ',(time()-tp))
    return xt,yt


x,y = dataset()
print(len(x))


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(len(x_train))
print(len(x_test))


#Decision Tree

t = time()
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)
print('time = ',(time()-t))

pred = clf.predict(x_test)
print(pred)
print('Accuracy = ',clf.score(x_test,y_test))

avg = 'weighted'
print('Precision score = ',metrics.precision_score(y_test, pred, average=avg))
print('Recall score = ',metrics.recall_score(y_test, pred, average=avg))
print('f1-score = ',metrics.f1_score(y_test, pred, average=avg))

c = 0
l = len(y_test)
for x in range (l):
    if pred[x]==y_test[x]:
        c += 1
print(c)
acc = (float)(c/l)
print('acc = ',acc)
print('time = ',(time()-t))


# # Support vector machine


t = time()
clf = SVC(kernel = 'poly')
print (clf)
clf.fit(x_train,y_train)
print('time = ',(time()-t))
pred = clf.predict(x_test)
print(pred)
print('Accuracy = ',clf.score(x_test,y_test))

avg = 'weighted'
print('Precision score = ',metrics.precision_score(y_test, pred, average=avg))
print('Recall score = ',metrics.recall_score(y_test, pred, average=avg))
print('f1-score = ',metrics.f1_score(y_test, pred, average=avg))

c = 0
l = len(y_test)
for x in range (l):
    if pred[x]==y_test[x]:
        c += 1
print(c)
acc = (float)(c/l)
print('acc = ',acc)
print('time = ',(time()-t))


# # K-Neighbors Classifier


t = time()
clf = KNeighborsClassifier()
print (clf)
clf.fit(x_train,y_train)
print('time = ',(time()-t))
pred = clf.predict(x_test)
print(pred)
print('Accuracy = ',clf.score(x_test,y_test))

avg = 'weighted'
print('Precision score = ',metrics.precision_score(y_test, pred, average=avg))
print('Recall score = ',metrics.recall_score(y_test, pred, average=avg))
print('f1-score = ',metrics.f1_score(y_test, pred, average=avg))


c = 0
l = len(y_test)
for x in range (l):
    if pred[x]==y_test[x]:
        c += 1
print(c)
acc = (float)(c/l)
print('acc = ',acc)
print('time = ',(time()-t))

