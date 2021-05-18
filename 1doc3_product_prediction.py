#%%
import pandas as pd 
import re
import numpy as np
import matplotlib.pyplot as plt 
import missingno as msno # para la visualización de los valores que faltan 

#%%
# Tenemos filas con más de una coma y por eso no lo lee bien. 

productos = pd.read_csv('Train.csv', header=None, names = ['label','title','is_validated_by_human'], error_bad_lines=False)
productos = productos.iloc[1:,:]

#%%
productos.head()
#%%
productos.shape
#%% Información acerca del dataset # en las columnas de title y "is_validated_by_human" tenemos valores nulos
productos.info() 

#%% Tenemos columnas mal anotadas y debemos corregir eso, porque toda la información del sample está en la misma columna 
productos['label'].value_counts()
productos['title'].value_counts()
productos['is_validated_by_human'].value_counts()
print(len(productos['is_validated_by_human']))
productos['label'].value_counts().plot(kind ='pie',figsize = (6,6))

#%%
lista_malos = []
for i in productos['label']:
    if len(i) > 10:
        lista_malos.append(i)
        
#%%
index = productos.index
for i in range(len(lista_malos)):
    condition = productos['label'] == lista_malos[i] # esto siempre se debe cumplir porque los de lista malos vienen del dataset original 
    condition_indices = index[condition].tolist()
    indice = condition_indices[0]
    title = re.findall(r'"([^"]*)"', lista_malos[i])
    if len(title) >= 1:
        label_and_validated = lista_malos[i].replace(title[0] ,'')
        label_and_validated = label_and_validated.split(',')
        if len(label_and_validated) >2:
            productos.at[indice,'title']= title
            productos.at[indice,'label']= label_and_validated[0]
            validated = label_and_validated[2]
            if 'YES' in validated:
                productos.at[indice,'is_validated_by_human'] = label_and_validated[2][0:3]
            elif  'NO' in validated:
                productos.at[indice,'is_validated_by_human'] = label_and_validated[2][0:2]
            elif validated == '4gb':
                productos.at[indice,'is_validated_by_human'] = 'YES'
        else:
            continue    
    else:
        continue    
    
#%% Para arreglar la parte de los labels , se crea una nueva columna llamada labels 
labels =[]
for label in productos['label']:
    l = label.split(',')[0]
    labels.append(l)
productos['labels'] = labels
productos['labels'].value_counts()

#%% TOCA QUITAR LOS NAN ANTES 
productos.dropna(inplace=True)
#%%
validateds = []
for val in productos['is_validated_by_human']:
    if 'NO' in val:
        validateds.append(val[0:2])
    elif 'YES' in val:
        validateds.append(val[0:3]) 
    elif  val == '4gb':
        validateds.append('YES')   

validateds.append('YES')
productos['is_validated_by_humans'] = validateds

#%%
productos.head()
#%% ELIMINAR LAS COLUMNAS ANTIGUAS

productos.drop(columns = ['label'],axis=1,inplace=True)

#%%
productos.drop(columns= ['is_validated_by_human'],axis=1,inplace=True)
#%% EN ESTE PUNTO YA SE ELIMINARON LAS COLUMNAS ANTIGUAS 
productos.head

#%%
print("la cantidad de valores nulos es:", productos.isnull().sum())

#%%
ax = productos['labels'].value_counts().plot(kind ='bar',figsize = (14,6))
ax.set_xlabel('Anotaciones')
ax.set_ylabel('Número de Anotaciones')
ax.set_title('Distribución del número de Anotaciones')

#%%
ax = productos['is_validated_by_humans'].value_counts().plot(kind ='bar',figsize = (14,6))
ax.set_xlabel('¿Es validado por humanos?')
ax.set_ylabel('Número de validaciones')
ax.set_title('Distribución de las validaciones')

#%% ESTADÍSTICAS DE LOS DATOS. 
print(productos['labels'].value_counts())
print(productos['is_validated_by_humans'].value_counts())

#%% EN ESTE PUNTO YA HABRÍA QUE IMPLEMENTAR LOS MODELOS 


