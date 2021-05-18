#%% Librerias utilizadas 
import pandas as pd 
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import missingno as msno 
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_selection import chi2
from sklearn import metrics

#%%
productos_val = pd.read_csv('Validation.csv', header=None, names = ['label','title','is_validated_by_human'], error_bad_lines=False,skiprows=1)
productos_val.head()

#%%
lista_malos = []
for i in productos_val['label']:
    if len(i) > 10:
        lista_malos.append(i)
        
#%%
index = productos_val.index
for i in range(len(lista_malos)):
    condition = productos_val['label'] == lista_malos[i] # esto siempre se debe cumplir porque los de lista malos vienen del dataset original 
    condition_indices = index[condition].tolist()
    indice = condition_indices[0]
    title = re.findall(r'"([^"]*)"', lista_malos[i])
    if len(title) >= 1:
        label_and_validated = lista_malos[i].replace(title[0] ,'')
        label_and_validated = label_and_validated.split(',')
        if len(label_and_validated) >2:
            productos_val.at[indice,'title']= title[0]
            productos_val.at[indice,'label']= label_and_validated[0]
            validated = label_and_validated[2]
            if 'YES' in validated:
                productos_val.at[indice,'is_validated_by_human'] = label_and_validated[2][0:3]
            elif  'NO' in validated:
                productos_val.at[indice,'is_validated_by_human'] = label_and_validated[2][0:2]
            elif validated == '4gb':
                productos_val.at[indice,'is_validated_by_human'] = 'YES'
        else:
            continue    
    else:
        continue    
    
#%% Para arreglar la parte de los labels , se crea una nueva columna llamada labels 
labels =[]
for label in productos_val['label']:
    l = label.split(',')[0]
    labels.append(l)
productos_val['labels'] = labels
productos_val['labels'].value_counts()

#%% AGREGAMOS LA MODA A LOS VALORES QUE FALTAN 
validated_mode = productos_val.is_validated_by_human.mode()[0]
productos_val.is_validated_by_human.fillna(validated_mode , inplace=True)

#%%
validateds = []
for val in productos_val['is_validated_by_human']:
    if 'NO' in val:
        validateds.append(val[0:2])
    elif 'YES' in val:
        validateds.append(val[0:3]) 
    elif  val == '4gb':
        validateds.append('YES')   

productos_val['is_validated_by_humans'] = validateds

#%%
productos_val.head()
#%% ELIMINAR LAS COLUMNAS ANTIGUAS

productos_val.drop(columns = ['label'],axis=1,inplace=True)

#%%
productos_val.drop(columns= ['is_validated_by_human'],axis=1,inplace=True)

#%%
productos_val['labels'].value_counts()
#%%
productos_val['is_validated_by_humans'].value_counts()