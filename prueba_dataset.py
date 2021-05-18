#%%
import pandas as pd 
import re
import numpy as np
import matplotlib.pyplot as plt 
import missingno as msno # para la visualización de los valores que faltan 
#%%
# Tenemos filas con más de una coma y por eso no lo lee bien. 

productos_train = pd.read_csv('Train.csv', header=None, names = ['label','title','is_validated_by_human'], error_bad_lines=False,skiprows=1)
productos_train.head()
#%%
productos_train.info() 
productos_train.iloc[2] # se accede así para las filas indexando como siempre desde 0 
#%% CON ILOC TOCA PASARLE UN NÚMERO
productos_train.iloc[:,1]
#%% CON LOC SE PASA EL NOMBRE DE LA COLUMNA 
productos_train.loc[:,'title'] 
#%%
productos_train.title
#%%
productos_train['title']