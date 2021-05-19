#%%
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
#%% DATOS DE ENTRENAMIENTO
train_data = pd.read_csv('new_train.csv')
train_data.head()
#%% DATOS DE VALIDACIÓN 
val_data = pd.read_csv('new_val.csv')
val_data.head() 
#%% PARA EL CONJUNTO DE TRAIN 
col = ['labels','title']
train_data = train_data[col]
train_data.columns = ['labels','title']
train_data['labels_id'] = train_data['labels'].factorize()[0] # pasamos los labels a números del 0 al 3 

labels_id_train_data = train_data[['labels','labels_id']].drop_duplicates().sort_values('labels_id')
labels_to_id = dict(labels_id_train_data.values)
id_to_labels = dict(labels_id_train_data[['labels_id','labels']].values)
train_data.head()

#%% PARA EL CONJUNTO DE VALIDACIÓN 
col_val = ['labels','title']
val_data = val_data[col_val]
val_data.columns = ['labels','title']
val_data['labels_id'] = val_data['labels'].factorize()[0] # pasamos los labels a números del 0 al 3 

labels_id_val_data = val_data[['labels','labels_id']].drop_duplicates().sort_values('labels_id')
labels_to_id_val = dict(labels_id_val_data.values)
id_to_labels_val = dict(labels_id_val_data[['labels_id','labels']].values)
val_data.head()

#%% PARA EL CONJUNTO DE ENTRENAMIENTO 
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm ='l2',encoding='latin-1',ngram_range = (1,2),stop_words = stopwords.words('spanish'))
features_train = tfidf.fit_transform(train_data.title).toarray()
decoded_labels_train = train_data.labels_id.astype(np.uint8) # esto es lo miso que labels_id ( los 4 labels pero en forma de número )
#%% PARA EL CONJUNTO DE VALIDACIÓN 
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm ='l2',encoding='latin-1',ngram_range = (1,2),stop_words = stopwords.words('spanish'))
features_val = tfidf.fit_transform(val_data.title).toarray()
decoded_labels_val = val_data.labels_id.astype(np.uint8) # esto es lo miso que labels_id ( los 4 labels pero en forma de número )
#%% INTENTO DE PASAR LOS DATOS DE VALIDACIÓN (SVM)

modelo = LinearSVC()
X_train = features_train[:,:546]
y_train = decoded_labels_train
X_test = features_val
y_test = decoded_labels_val
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_train_data.labels.values, yticklabels=labels_id_train_data.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(metrics.classification_report(y_test, y_pred, target_names=train_data['labels'].unique()))
print(" El accuracy para el modelo haciendo uso del SVM lineal es:",metrics.accuracy_score(y_test, y_pred))
print(" La precisión para el modelo haciendo uso del SVM lineal es:",metrics.precision_score(y_test, y_pred,average='weighted'))
print(" La cobertura para el modelo haciendo uso del SVM lineal es:",metrics.recall_score(y_test, y_pred,average='weighted'))
print(" El F1 score  para el modelo haciendo uso del SVM lineal es:",metrics.f1_score(y_test, y_pred,average='weighted'))

#%% INTENTO DE PASAR LOS DATOS DE VALIDACIÓN (MNB)
modelo2 = MultinomialNB()
X_train = features_train[:,:546]
y_train = decoded_labels_train
X_test = features_val
y_test = decoded_labels_val
modelo2.fit(X_train, y_train)
y_pred = modelo2.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_train_data.labels.values, yticklabels=labels_id_train_data.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(metrics.classification_report(y_test, y_pred, target_names=train_data['labels'].unique()))
print(" El accuracy para el modelo haciendo uso del  MNB es:",metrics.accuracy_score(y_test, y_pred))
print(" La precisión para el modelo haciendo uso del MNB es:",metrics.precision_score(y_test, y_pred,average='weighted'))
print(" La cobertura para el modelo haciendo uso del MNB es:",metrics.recall_score(y_test, y_pred,average='weighted'))
print(" El F1 score  para el modelo haciendo uso del MNB es:",metrics.f1_score(y_test, y_pred,average='weighted'))

#%% NAIVE BAYES MULTINOMIAL 
X_train = train_data['title']
y_train = train_data['labels']
X_test = val_data['title']
y_test = val_data['labels']
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])

nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_train_data.labels.values, yticklabels=labels_id_train_data.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(metrics.classification_report(y_test, y_pred, target_names=train_data['labels'].unique()))
print(" El accuracy para el modelo haciendo uso del  MNB es:",metrics.accuracy_score(y_test, y_pred))
print(" La precisión para el modelo haciendo uso del MNB es:",metrics.precision_score(y_test, y_pred,average='weighted'))
print(" La cobertura para el modelo haciendo uso del MNB es:",metrics.recall_score(y_test, y_pred,average='weighted'))
print(" El F1 score  para el modelo haciendo uso del MNB es:",metrics.f1_score(y_test, y_pred,average='weighted'))


#%% CLASIFICADOR CON DECSENSO DEL GRADIENTE 

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_train_data.labels.values, yticklabels=labels_id_train_data.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(metrics.classification_report(y_test, y_pred, target_names=train_data['labels'].unique()))
print(" El accuracy para el modelo haciendo uso del  SGDC es:",metrics.accuracy_score(y_test, y_pred))
print(" La precisión para el modelo haciendo uso del SGDC es:",metrics.precision_score(y_test, y_pred,average='weighted'))
print(" La cobertura para el modelo haciendo uso del SDGC es:",metrics.recall_score(y_test, y_pred,average='weighted'))
print(" El F1 score  para el modelo haciendo uso del SGDC es:",metrics.f1_score(y_test, y_pred,average='weighted'))

#%% REGRESIÓN LOGÍSTICA 
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_train_data.labels.values, yticklabels=labels_id_train_data.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(metrics.classification_report(y_test, y_pred, target_names=train_data['labels'].unique()))
print(" El accuracy para el modelo haciendo uso de regresión logística es:",metrics.accuracy_score(y_test, y_pred))
print(" La precisión para el modelo haciendo uso de regresión logística es:",metrics.precision_score(y_test, y_pred,average='weighted'))
print(" La cobertura para el modelo haciendo uso de regresión logística es:",metrics.recall_score(y_test, y_pred,average='weighted'))
print(" El F1 score  para el modelo haciendo uso de regresión logística es:",metrics.f1_score(y_test, y_pred,average='weighted'))

#%% XGBOOST 
X_train = train_data['title']
y_train = train_data['labels']
X_test = val_data['title']
y_test = val_data['labels']
clf_xgb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', XGBClassifier( objective = 'reg:logistic',random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)),
              ])
clf_xgb.fit(X_train, y_train,clf__verbose = True,clf__eval_metric = 'aucpr')
y_pred = clf_xgb.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_train_data.labels.values, yticklabels=labels_id_train_data.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(metrics.classification_report(y_test, y_pred, target_names=train_data['labels'].unique()))
print(" El accuracy para el modelo haciendo uso de XGBOOST es:",metrics.accuracy_score(y_test, y_pred))

#%% OPTIMIZACION DEL XGBOOST 

param_grid = {'max_depth':[3,4,5],
              'learning_rate':[0.1,0.01,0.05],
              'gamma':[0,0.25,1],
              'reg_lambda':[0,1.0,10.0]
              }

optimal_params = GridSearchCV(estimator=XGBClassifier( objective = 'reg:logistic',random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7),
                              param_grid= param_grid,
                              scoring='roc_auc',
                              verbose = 2, # 2 para ver que es lo que está haciendo esto
                              n_jobs =10,
                              cv = 3
                               )
#%%
optimal_params.fit(X_train, y_train)

# gamma = 0.25, lr = 0.05 max_depth = 5 reg_lambda = 1.0
#%% XGBOOST CON LOS PARÁMETROS  YA OPTIMIZADOS 

X_train = train_data['title']
y_train = train_data['labels']
X_test = val_data['title']
y_test = val_data['labels']
clf_xgb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', XGBClassifier( random_state=42,objective = 'reg:logistic',learn_rate = 0.05, gamma = 0.25, max_depth = 5, reg_lambda = 1, scale_pos_weight = 3, seed=2, colsample_bytree=0.6, subsample=0.7)),
              ])
clf_xgb.fit(X_train, y_train,clf__verbose = True,clf__eval_metric = 'aucpr')
y_pred = clf_xgb.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_id_train_data.labels.values, yticklabels=labels_id_train_data.labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(metrics.classification_report(y_test, y_pred, target_names=train_data['labels'].unique()))
print(" El accuracy para el modelo haciendo uso de XGBOOST es:",metrics.accuracy_score(y_test, y_pred))

#%% mostrando el primer árbol 
X_train = train_data['title']
y_train = train_data['labels']
X_test = val_data['title']
y_test = val_data['labels']
clf_xgb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', XGBClassifier( random_state=42,objective = 'reg:logistic',learn_rate = 0.05, gamma = 0.25, max_depth = 5, reg_lambda = 1, scale_pos_weight = 3, seed=2, colsample_bytree=0.6, subsample=0.7, n_estimators = 1)),
              ])
clf_xgb.fit(X_train, y_train,clf__verbose = True,clf__eval_metric = 'aucpr')
y_pred = clf_xgb.predict(X_test)

#%%
node_params = {'shape':'box',
               'style': 'filled,rounded',
               'fillcolor':'#78cbe'
               }

leaf_params = {'shape':'box',
               'style': 'filled',
               'fillcolor':'#e48038'
               }
               
xgb.to_graphviz(clf_xgb,num_trees = 0, size = "10,10",
                 condition_node_params = node_params,
                 leaf_node_params = leaf_params)               
