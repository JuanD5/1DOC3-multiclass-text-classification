#%% ITERAR PARA ARREGLAR LAS ANOTACIONES 

labels =[]
for label in productos['label']:
    temp = label.split(',')[0]
    labels.append(temp)
productos['labels'] = labels
productos.head()



#%% Visualización de los valores que faltan 
print(productos.isnull().sum())
msno.bar(productos)
#%% MÉTODO PARA ELMININAR LOS VALORES NULOS 
productos.dropna(inplace=True)
print("la cantidad de valores nulos es:", productos.isnull().sum())
msno.bar(productos)
# %%
productos['labels'].value_counts().plot(kind ='pie',figsize = (6,6))
#%%
ax = productos['labels'].value_counts().plot(kind ='bar',figsize = (14,6))
ax.set_xlabel('Anotaciones')
ax.set_ylabel('Número de Anotaciones')
ax.set_title('Distribución del número de Anotaciones')

#%%
validated = []
for val in productos['is_validated_by_human']:
    if 'NO' in val:
        validated.append(val[0:2])
    elif 'YES' in val:
        validated.append(val[0:3]) 
    elif  val == '4gb':
        validated.append('YES')   

validated.append('YES')
productos['is_validated_by_humans'] = validated
productos.head()

productos.drop(columns = ['label'],axis=1,inplace=True)
productos.drop(columns = ['is_validated_by_human'],axis=1,inplace=True)
productos.head()