#%% ITERAR PARA ARREGLAR LAS ANOTACIONES 

labels =[]
for label in productos_train['label']:
    temp = label.split(',')[0]
    labels.append(temp)
productos_train['labels'] = labels
productos_train.head()



#%% Visualización de los valores que faltan 
print(productos_train.isnull().sum())
msno.bar(productos_train)
#%% MÉTODO PARA ELMININAR LOS VALORES NULOS 
productos_train.dropna(inplace=True)
print("la cantidad de valores nulos es:", productos_train.isnull().sum())
msno.bar(productos_train)
# %%
productos_train['labels'].value_counts().plot(kind ='pie',figsize = (6,6))
#%%
ax = productos_train['labels'].value_counts().plot(kind ='bar',figsize = (14,6))
ax.set_xlabel('Anotaciones')
ax.set_ylabel('Número de Anotaciones')
ax.set_title('Distribución del número de Anotaciones')

#%%
validated = []
for val in productos_train['is_validated_by_human']:
    if 'NO' in val:
        validated.append(val[0:2])
    elif 'YES' in val:
        validated.append(val[0:3]) 
    elif  val == '4gb':
        validated.append('YES')   

validated.append('YES')
productos_train['is_validated_by_humans'] = validated
productos_train.head()

productos_train.drop(columns = ['label'],axis=1,inplace=True)
productos_train.drop(columns = ['is_validated_by_human'],axis=1,inplace=True)
productos_train.head()