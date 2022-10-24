from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def f1_score_(y_proba,y_test):
  proba = y_proba[:,1] >= 0.3
  proba = proba.astype(np.int) 
  return f1_score( proba,y_test) 

def train_model(X_train, y_train, X_test=None, y_test=None):
    k=[3]
    accuracy_train=[]
    accuracy_test=[]
    metadata = {}

    for i in tqdm(k):
        model=KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_train)
        acc=accuracy_score(y_pred,y_train)

        print('for k=',i,'Accuracy Score',acc)

        accuracy_train.append(acc)
        y_proba=model.predict_proba(X_train)
        f1_scor_train=f1_score_(y_proba,y_train)

        print('for k=',i,'f1 score ',f1_scor_train)

    if X_test is not None and y_test is not None:
        for i in tqdm(k):
            model=KNeighborsClassifier(n_neighbors=i)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            acc=accuracy_score(y_pred,y_test)

            print('for k=',i,'Accuracy Score',acc)

            accuracy_test.append(acc)
            y_proba=model.predict_proba(X_test)
            f1_scor_test=f1_score_(y_proba,y_test)

            print('for k=',i,'f1 score ',f1_scor_test)
    
    metadata["accuracy_train"] = accuracy_train
    metadata["accuracy_test"] = accuracy_test
    metadata["f1_scor_train"] = f1_scor_train
    metadata["f1_scor_test"] = f1_scor_test
    metadata["y_pred"] = y_pred
  
    return model, metadata


