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
#%% EN ESTE PUNTO YA SE ELIMINARON LAS COLUMNAS ANTIGUAS 
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

#%% PREPARACIÓN DEL NUEVO DATASET 

col = ['labels','title']
productos_train = productos_train[col]
productos_train.columns = ['labels','title']
productos_train['labels_id'] = productos_train['labels'].factorize()[0]
labels_id_productos_train = productos_train[['labels','labels_id']].drop_duplicates().sort_values('labels_id')
labels_to_id = dict(labels_id_productos_train.values)
id_to_labels = dict(labels_id_productos_train[['labels_id','labels']].values)
productos_train.head()

#%% TRATAMIENTO DEL DESBALANCEO DE LOS DATOS 
fig = plt.figure(figsize = (8,6))
productos_train.groupby('labels').title.count().plot.bar(ylim=0)
plt.show()

#%%
nltk.download('stopwords')


#%%
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm ='l2',encoding='latin-1',ngram_range = (1,2),stop_words = stopwords.words('spanish'))
features = tfidf.fit_transform(productos_train.title).toarray()
new_labels = productos_train.labels_id.astype(np.uint8)
#%%
features.shape

#%% los unigramas y bigramas mas relacionados con cada una de las anotaciones 
N = 2
for Product, category_id in sorted(labels_to_id.items()):
  features_chi2 = chi2(features, new_labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


#%% NAIVE BAYES CLASSIFIER MULTINOMIAL 

X_train, X_test, y_train, y_test = train_test_split(productos_train['title'], productos_train['labels'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

#%%
print(clf.predict(count_vect.transform(["ipad mini en excelente estado."])))

#%%
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, new_labels, productos_train.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_productos_train.labels.values, yticklabels=labels_id_productos_train.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#%%
print(metrics.classification_report(y_test, y_pred, target_names=productos_train['labels'].unique()))
print(" El accuracy para el modelo haciendo uso del SVM lineal es:",metrics.accuracy_score(y_test, y_pred))