#%% Librerias utilizadas 
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import missingno as msno 
import xgboost as xgb 
import gensim
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
from bs4 import BeautifulSoup
from numpy import random
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
#%% LECTURA DE LOS ARCHIVOS 
productos_train = pd.read_csv('Train.csv', header=None, names = ['label','title','is_validated_by_human'], error_bad_lines=False,skiprows=1)
productos_train.head()

#%% Dimensiones del dataframe 
productos_train.shape
#%% Información acerca del dataset # en las columnas de title y "is_validated_by_human" tenemos valores nulos
productos_train.info() 

#%% Tenemos columnas mal anotadas y debemos corregir eso, porque toda la información del sample está en la misma columna 
productos_train['label'].value_counts()
productos_train['title'].value_counts()
productos_train['is_validated_by_human'].value_counts()
#%%
lista_malos = []
for i in productos_train['label']:
    if len(i) > 10:
        lista_malos.append(i)
        
#%%
index = productos_train.index
for i in range(len(lista_malos)):
    condition = productos_train['label'] == lista_malos[i] # esto siempre se debe cumplir porque los de lista malos vienen del dataset original 
    condition_indices = index[condition].tolist()
    indice = condition_indices[0]
    title = re.findall(r'"([^"]*)"', lista_malos[i])
    if len(title) >= 1:
        label_and_validated = lista_malos[i].replace(title[0] ,'')
        label_and_validated = label_and_validated.split(',')
        if len(label_and_validated) >2:
            productos_train.at[indice,'title']= title[0]
            productos_train.at[indice,'label']= label_and_validated[0]
            validated = label_and_validated[2]
            if 'YES' in validated:
                productos_train.at[indice,'is_validated_by_human'] = label_and_validated[2][0:3]
            elif  'NO' in validated:
                productos_train.at[indice,'is_validated_by_human'] = label_and_validated[2][0:2]
            elif validated == '4gb':
                productos_train.at[indice,'is_validated_by_human'] = 'YES'
        else:
            continue    
    else:
        continue    
    
#%% Para arreglar la parte de los labels , se crea una nueva columna llamada labels 
labels =[]
for label in productos_train['label']:
    l = label.split(',')[0]
    labels.append(l)
productos_train['labels'] = labels
productos_train['labels'].value_counts()

#%% AGREGAMOS LA MODA A LOS VALORES QUE FALTAN 
validated_mode = productos_train.is_validated_by_human.mode()[0]
productos_train.is_validated_by_human.fillna(validated_mode , inplace=True)

#%%
validateds = []
for val in productos_train['is_validated_by_human']:
    if 'NO' in val:
        validateds.append(val[0:2])
    elif 'YES' in val:
        validateds.append(val[0:3]) 
    elif  val == '4gb':
        validateds.append('YES')   

validateds.append('YES')
productos_train['is_validated_by_humans'] = validateds

#%%
productos_train.head()
#%% ELIMINAR LAS COLUMNAS ANTIGUAS

productos_train.drop(columns = ['label'],axis=1,inplace=True)

#%%
productos_train.drop(columns= ['is_validated_by_human'],axis=1,inplace=True)

#%%Limpieza del texto 
nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('spanish'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "html.parser").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
#%%
productos_train['title'] = productos_train['title'].apply(clean_text)

#%%
productos_train = productos_train[productos_train['title'].notna()]

#%%
productos_train.to_csv('new_train.csv', encoding='utf-8', index=False)
#%% EN ESTE PUNTO YA SE ELIMINARON LAS COLUMNAS ANTIGUAS  Y QUEDARON LIMPIOS LOS DATOS 
productos_train.head()

#%%
print("la cantidad de valores nulos es:", productos_train.isnull().sum())

#%%
ax = productos_train['labels'].value_counts().plot(kind ='bar',figsize = (14,6))
ax.set_xlabel('Anotaciones')
ax.set_ylabel('Número de Anotaciones')
ax.set_title('Distribución del número de Anotaciones')

#%%
ax = productos_train['is_validated_by_humans'].value_counts().plot(kind ='bar',figsize = (14,6))
ax.set_xlabel('¿Es validado por humanos?')
ax.set_ylabel('Número de validaciones')
ax.set_title('Distribución de las validaciones')

#%% ESTADÍSTICAS DE LOS DATOS. 
print(productos_train['labels'].value_counts())
print(productos_train['is_validated_by_humans'].value_counts())

#%%
productos_train.labels.value_counts(normalize = True)
productos_train.labels.value_counts(normalize = True).plot.barh()
plt.show()

#%% 0 es NO , 1 es YES 
productos_train['is_validated_by_humans'] = np.where(productos_train.is_validated_by_humans == 'YES',1,0)
#%%
productos_train.is_validated_by_humans.value_counts()
#%% DE ESTA GRÁFICA SE INFIERE QUE LA MAYORÍA DE ANOTACIONES QUE FUERON VALIDADAS POR HUMANOS SON LAS DE NOTEBOOKS Y TABLETS y LAS QUE EN GRAN MAYORÍA FUERON VALIDADAS POR MÁQUINAS SON LAS DE TELEPHONES Y CELLPHONES. 
productos_train.groupby('labels')['is_validated_by_humans'].mean().plot.bar()
plt.ylabel('Porcentaje de anotaciones')
plt.title('Distribución de las anotaciones validadas por humanos')
plt.show()