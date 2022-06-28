
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pickle


fh = open("scal_x.pkl","br")
escal_x = pickle.load(fh)
fh.close()

fh = open("scal_y.pkl","br")
escal_y = pickle.load(fh)
fh.close()

fh = open("Regresion1.pkl","br")
reg1 = pickle.load(fh)
fh.close()

fh = open("Regresion2.pkl","br")
reg2 = pickle.load(fh)
fh.close()

fh = open("RandomForest.pkl","br")
clf = pickle.load(fh)
fh.close()

fh = open("dics.pkl","br")
list_dic = pickle.load(fh)
fh.close()

data = pd.read_csv("datos_prueba.csv")
data

del data["MODELYEAR"] 
del data["MODEL"]

dato = data.iloc[2,:-2]

dato2 = []
for i in dato.index:
    dato2.append(input("Dame el valor de {} :".format(i)))

dato =  np.array([float(i) for i in dato2])

if clf.predict(dato.reshape(1,-1)) == 0:
    consumo = escal_x.transform(np.array([[dato[7]]]))
    emision = reg1.predict(consumo)
else:
    consumo = escal_x.transform(np.array([[dato[7]]]))
    emision = reg2.predict(consumo)
    
emision = escal_y.inverse_transform(emision)

print("La emision del auto es: ",emision[0][0])