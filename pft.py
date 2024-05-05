# %% [markdown]
# El objetivo del presente Proyecto es apoyar al operador de telecomunicaciones Interconnect a quien le gustaría poder pronosticar su tasa de cancelación de clientes. Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. 

# %% [markdown]
# El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.
# ### Servicios de Interconnect
# 
# Interconnect proporciona principalmente dos tipos de servicios:
# 
# 1. Comunicación por teléfono fijo. El teléfono se puede conectar a varias líneas de manera simultánea.
# 2. Internet. La red se puede configurar a través de una línea telefónica (DSL, *línea de abonado digital*) o a través de un cable de fibra óptica.
# 
# 
# Algunos otros servicios que ofrece la empresa incluyen:
# 
# - Seguridad en Internet: software antivirus (*ProtecciónDeDispositivo*) y un bloqueador de sitios web maliciosos (*SeguridadEnLínea*).
# - Una línea de soporte técnico (*SoporteTécnico*).
# - Almacenamiento de archivos en la nube y backup de datos (*BackupOnline*).
# - Streaming de TV (*StreamingTV*) y directorio de películas (*StreamingPelículas*)
# 
# La clientela puede elegir entre un pago mensual o firmar un contrato de 1 o 2 años. Puede utilizar varios métodos de pago y recibir una factura electrónica después de una transacción.
# 
# ###
# 

# %% [markdown]
# ### Descripción de los datos
# 
# Los datos consisten en archivos obtenidos de diferentes fuentes:
# 
# - `contract.csv` — información del contrato;
# - `personal.csv` — datos personales del cliente;
# - `internet.csv` — información sobre los servicios de Internet;
# - `phone.csv` — información sobre los servicios telefónicos.
# 
# En cada archivo, la columna `customerID` (ID de cliente) contiene un código único asignado a cada cliente. La información del contrato es válida a partir del 1 de febrero de 2020.

# %% [markdown]
# # Aclaración: Resumen
# 
# Característica objetivo: la columna `'EndDate'` es igual a `'No'`.
# 
# Métrica principal: AUC-ROC.
# 
# Métrica adicional: exactitud.
# 
# Criterios de evaluación:
# 
# - AUC-ROC < 0.75 — 0 SP
# - 0.75 ≤ AUC-ROC < 0.81 — 4 SP
# - 0.81 ≤ AUC-ROC < 0.85 — 4.5 SP
# - 0.85 ≤ AUC-ROC < 0.87 — 5 SP
# - 0.87 ≤ AUC-ROC < 0.88 — 5.5 SP
# - AUC-ROC ≥ 0.88 — 6 SP

# %% [markdown]
# - **Etapas del proyecto:**
#     - Hacer un plan de trabajo.
#     - Investigar la tarea.
#     - Desarrollar un modelo.
#     - Preparar el informe.

# %% [markdown]
# # PLAN DE TRABAJO 

# %% [markdown]
# El primer paso a realizar es el Análisis Exploratorio de Datos (EDA, por sus siglas en inglés) es una etapa crucial en el proceso de análisis de datos que implica explorar, resumir y visualizar los datos para comprender mejor sus características. 
# 1. Se realizará un Resumen estadístico inicial: Esto implica calcular estadísticas descriptivas básicas como la media, la mediana, la desviación estándar, el rango, los cuartiles, etc., para cada variable en el conjunto de datos. Esto proporciona una comprensión inicial de la distribución y la variabilidad de los datos.
# 
# 2. Tratamiento de datos faltantes: Identificar y manejar los valores faltantes en el conjunto de datos. Esto podría implicar imputar valores, eliminar registros o variables con valores faltantes, o utilizar técnicas más avanzadas como la imputación multivariable.
# 
# 3.Exploración de la distribución de las variables: Visualizar la distribución de las variables utilizando histogramas, diagramas de caja (boxplots), o gráficos de densidad para comprender la forma y la dispersión de los datos.
# 
# 4. Análisis de correlación: Evaluar la relación entre variables mediante el cálculo de correlaciones y la creación de mapas de calor (heatmaps) de correlación. Esto puede ayudar a identificar posibles relaciones lineales o no lineales entre las variables.
# 
# 5.Análisis de outliers: Identificar valores atípicos que podrían ser errores de medición o indicativos de comportamiento inusual en los datos. Esto puede hacerse visualmente mediante diagramas de dispersión (scatterplots) o cuantitativamente utilizando técnicas como el método del rango intercuartil (IQR) o el Z-score.
# 
# 6.Exploración de variables categóricas: Si el conjunto de datos incluye variables categóricas, es importante explorar su distribución y relación con otras variables a través de tablas de frecuencia y gráficos de barras en caso necesario se transformaran en este mismo momento en variables numéricas antes de iniciar a trabajar con los modelos.
# 
# 

# %% [markdown]
# En el informe, responde las siguientes preguntas:
# 
# - ¿Qué pasos del plan se realizaron y qué pasos se omitieron (explica por qué)?
# - ¿Qué dificultades encontraste y cómo lograste resolverlas?
# - ¿Cuáles fueron algunos de los pasos clave para resolver la tarea?
# - ¿Cuál es tu modelo final y qué nivel de calidad tiene?
# 
# Estos son algunos de los criterios utilizados por el líder del equipo:
# 
# - ¿Respondiste todas las preguntas y con respuestas claras?

# %% [markdown]
# # Inicio
# Carga de datos

# %%
import math # TODO Librería no utilizada
import itertools # TODO Librería no utilizada
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # TODO Librería no utilizada
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# %%
try:
    df_contract = pd.read_csv('/home/rosypc2/Escritorio/PROYECTO FINAL/contract.csv') # TODO No uses rutas absolutas, solo relativas a la carpeta del proyecto
    df_personal = pd.read_csv('/home/rosypc2/Escritorio/PROYECTO FINAL/personal.csv')
    df_internet = pd.read_csv('/home/rosypc2/Escritorio/PROYECTO FINAL/internet.csv')
    df_phone = pd.read_csv('/home/rosypc2/Escritorio/PROYECTO FINAL/phone.csv')                

except:
    df_contract = pd.read_csv('../datasets/final_provider/contract.csv')
    df_personal = pd.read_csv('../datasets/final_provider/personal.csv')
    df_internet = pd.read_csv('../datasets/final_provider/internet.csv')
    df_phone = pd.read_csv('../datasets/final_provider/phone.csv')

# %% [markdown]
# # Analisis Exploratorio de datos (EDA)
# En este apartado realizare un analisis de cada uno de los dataframe que se proporcionaron, realizare las transformaciones necesarias para poder trabajar sin problemas en los modelos

# %%
df_contract.info()

# %%
df_contract.head(5)

# %%
df_contract.describe()

# %%
# Creamos la grafica para visualizar la distribución de la variable MonthlyCharges
sns.displot(df_contract['MonthlyCharges'], kde=True)
plt.show()

# %% [markdown]
# Primeras observaciones, en este dataframe tenemos dos columnas que nos pueden ayudar a determinar la antiguedad del cliente en la compañia, al restar la fecha en que abandono los servicios de la fecha en que inicio, previo a esto es necesario cambiar el tipo de variable de object a datetime, Ademas es importante observar que nuestra variable objetivo se ubica cuando en la columna "EndDate" se observa un valor de No, lo que nos indica que el cliente no ha dejado la compañia, es conveniente cambiar el nombre de esta columna por el de "Exited" para identificar claramente a nuestra variable objetivo. 
# Tambien observamos que la variable MonthlyChargues tiene una distribucion en la que el mayor numero de clientes tiene un gasto mensual de 20, a pesar de que el valor promedio es de 64.7
# 

# %%
# Reemplazamos todos los valores 'No' en EndDate column with None, para poder calcular los meses de antiguedad en la empresa y convertir a datetime
df_contract['EndDate'] = np.where(df_contract['EndDate'] == 'No', None, df_contract['EndDate'])

# Cambiamos el tipo de variable
df_contract['BeginDate'] = pd.to_datetime(df_contract['BeginDate'], format='%Y-%m-%d')
df_contract['EndDate'] = pd.to_datetime(df_contract['EndDate'], format='%Y-%m-%d')

# Creamos la columna con el número de meses que el cliente ha estado en la compañia
df_contract['MonthsInCompany'] = (df_contract['EndDate'] - df_contract['BeginDate']) / pd.Timedelta(days=30)

# Realizacmos la sustracción y dividimos en el lapso de 30 dias para determinar los meses
df_contract['MonthsInCompany'].fillna((df_contract['EndDate'].max() - df_contract['BeginDate']) / pd.Timedelta(days=30), inplace=True)

# Creamos la columna en donde observamos nuesta variable objetivo
df_contract['Exited'] = df_contract['EndDate'].notna().astype('uint8')

# Eliminamos las columnas que ya no necesitamos
df_contract.drop(['BeginDate', 'EndDate'], axis=1, inplace=True)

# %%
df_contract

# %% [markdown]
# En virtud de que tenemos variables categoricas que nos van a entorpecer el uso de los modelos podemos desde este momento transformar a variables numéricas, en este caso son las variables PaperlessBilling  a la que le aplicare OHE y a las variables Type y PaymentMethod las transformare usando Label Encoding, posteriormente se eliminaran las columans originales.

# %%
class_counts = df_contract['Exited'].value_counts()

print(class_counts)

sns.countplot(x='Exited', data=df_contract)
plt.show()

# %% [markdown]
# Claramente hay un desequilibiro de clases, situación que hay que tomar en cuenta al momento de desarrollar el modelo.

# %%
df_contract = pd.get_dummies(df_contract, columns=['PaperlessBilling'], drop_first=True)


# %%

label_encoder= LabelEncoder()
df_contract["Type_Encoded"]=label_encoder.fit_transform(df_contract['Type'])
df_contract


# %%
label_encoder= LabelEncoder()
df_contract["PaymentMethod_Encoded"]=label_encoder.fit_transform(df_contract['PaymentMethod'])
df_contract


# %% [markdown]
# 

# %%
df_contract.drop(['Type', 'PaymentMethod'] , axis=1, inplace=True)
df_contract

# %% [markdown]
# Ahora analizaremos el df_personal y realizaremos los cambios pertinentes

# %%
df_personal.info()

# %%
df_personal.head()

# %% [markdown]
# En este caso se considera indispensable transformar las variables categoricas gender, Partner y Dependents en variables numéricas y en los tres casos lo realizaré con OHE

# %%
df_personal = pd.get_dummies(df_personal, columns=['gender'], drop_first=True)
df_personal = pd.get_dummies(df_personal, columns=['Partner'], drop_first=True)
df_personal = pd.get_dummies(df_personal, columns=['Dependents'], drop_first=True)

df_personal

# %% [markdown]
# Continuamos analizando y modificando el df_internet

# %%
df_internet.info()

# %%
df_internet.head

# %% [markdown]
# Observamos que en este dataframe, todas las variables son categoricas y podriamso convertirlas a valores boleanos, pero considero mejor convertiras en valores numericos utilizando OHE, exceptuado el  customerID que posteriormente nos servira de base para unir nuestros dataframes en uno solo.

# %%
df_internet = pd.get_dummies(df_internet, columns =['InternetService'], drop_first=True)
df_internet = pd.get_dummies(df_internet, columns=['OnlineSecurity'], drop_first=True)
df_internet = pd.get_dummies(df_internet, columns =['OnlineBackup'], drop_first=True)
df_internet = pd.get_dummies(df_internet, columns=['DeviceProtection'], drop_first=True)
df_internet = pd.get_dummies(df_internet, columns =['TechSupport'], drop_first=True)
df_internet = pd.get_dummies(df_internet, columns=['StreamingTV'], drop_first=True)
df_internet = pd.get_dummies(df_internet, columns =['StreamingMovies'], drop_first=True)
df_internet                                      

# %% [markdown]
# Ahora vamos a analizar el df_phone

# %%
df_phone.info()

# %%
df_phone.head()

# %% [markdown]
# En este dataframe lo unico que se debe transformar la columna MultipleLines utilizando OHE

# %%
df_phone = pd.get_dummies(df_phone, columns =['MultipleLines'], drop_first=True)
df_phone                                     

# %% [markdown]
# Despues de esta primera revision de los datos podemos determinar que no tenemos datos nulos en ninguno de los dataframes que nos proporcionaron, asimismo existen dos dataframes que tienen el mismo numero de filas que son el que contiene la información de contrato y la que contiene los datos personales de los clintes (df_contact y df_personal), las que se refieren a los servicios tanto de internet como de telefonia son diferentes entre si y tambien con respecto a los otros. 
# Tenemos la variable "customerID presente en las 4 tablas, antes de proceder unificar la información en una sola tabla, es conveniente hacer los cambios pertinentes y analizar algunos graficos para tener mayor claridad en la importancia de cada una de las variables.

# %%
# Merge all DataFrames into a single DataFrame on the customerID column
df = pd.merge(df_contract, df_personal, on='customerID', how='outer')
df = pd.merge(df, df_internet, on='customerID', how='outer')
df = pd.merge(df, df_phone, on='customerID', how='outer')

# Drop customerID column as it is no longer needed
df = df.drop('customerID', axis=1).reset_index(drop=True)

# %%
df.info()

# %% [markdown]
# De esta informacion podemos decir que de la columna 12 a la 18 tenemos missing values, los podemos sustituir por "0" y convertir dichas columnas a valores numéricos. 

# %%
df_filled = df.fillna(0)
df_filled

# %%
df_filled.info()

# %%
df = df_filled

# %%
# Calcular la matriz de correlación
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

# %% [markdown]
# En esta grafica podemos observar que hay variables que se relacionan entre si con un valor alto de correlación como es el servicio de internet con fibra optica  esta directamente relacionado con los carrgos mensuales, y estos a su vez se relaciona fuertemente con la variable Type,asi como con el Type. Asimismo la variable partner se relaciona con la variable dependent, la variable monthlycharges tambien se relaciona con los servicios de streaming que el cliente contrata. Lamentablemente la relacion directa con nuestra varaible objetivo no se aprecia tan claramente. 
# Por lo anterior realizare otra exploración para averiaguar con que variables existe una mayor relación

# %%
# Seleccionar la fila o columna correspondiente a 'Exited'
correlation_with_Exited = correlation_matrix['Exited']  # Si 'Exited' es una columna
# O
correlation_with_Exited = correlation_matrix.loc['Exited']  # Si 'Exited' es una fila

# Ordenar los valores de correlación
sorted_correlation = correlation_with_Exited.sort_values(ascending=False)

# Crear un nuevo mapa de calor solo con las variables ordenadas
sorted_correlation_matrix = correlation_matrix.loc[sorted_correlation.index, sorted_correlation.index]

# Crear la figura con un tamaño específico
plt.figure(figsize=(10, 8))

# Visualizar la matriz de correlación ordenada como un mapa de calor
sns.heatmap(sorted_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Agregar título
plt.title('Matriz de Correlación Ordenada con respecto a Exited')

# Mostrar el mapa de calor
plt.show()


# %%
# Convertir el diccionario a un DataFrame
df_resultados = pd.DataFrame(sorted_correlation_matrix)

# Imprimir el DataFrame
print(df_resultados)

# %% [markdown]
# Como se puede apreciar en los resultados anteriores, la correlacion de la variable objetivo "Exited" con las demas variables no presenta valores muy altos, sin embargo se pueden determinar variables en las que la correlación es menor a cero, lo cual me indica que no existe influencia de ellas en que el cliente decida dejar la compañia, en consecuencia, tomare la decision de eliminarlas del dataframe para poder desarrollar el modelo.

# %%
# Drop the following columns as they are not relevant to the prediction of Churn or they are highly correlated with other columns
cols = ["gender_Male",                  
"StreamingTV_Yes",             
"StreamingMovies_Yes",          
"Partner_Yes",                  
"Dependents_Yes",              
"DeviceProtection_Yes",          
"OnlineBackup_Yes",              
"TechSupport_Yes",               
"OnlineSecurity_Yes",             
"MonthsInCompany",                
"Type_Encoded",                
]

df.drop(cols, axis=1, inplace=True)

# Print the general/summary information about the DataFrames
df.info()

# Print a random sample of 5 rows from the DataFrame
df.sample(5)


# %% [markdown]
# Con este resultado tambien considero necesario eliminar la columna Total Charges en primer lugar porque esta definida como un object y en segundo lugar porque los cargos mensuales nos proporcionan la información que se requiere para el proceso de modelado. Asimismo es necesario estandarizar todos los valores numéricos. Utilizare StandardScaler

# %%
df= df.drop('TotalCharges',axis=1)

# %%
df = pd.get_dummies(df, columns=['Exited'], drop_first=True)


# %%
# Creando el objeto 
scaler = StandardScaler()

# Aplicar el escalado estándar a todas las columnas del DataFrame
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Persisitiendo los cambios en el df
df=df_scaled
df

# %% [markdown]
# # CONSTRUCCION DEL MODELO IDEAL

# %%
# Adjust the class imbalance by oversampling the minority class
df_majority = df[df['Exited_1'] == 0]
df_minority = df[df['Exited_1'] == 1]
df_minority_upsampled = df_minority.sample(df_majority.shape[0], replace=True,  random_state=12345)
df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=12345).reset_index(drop=True)



# %% [markdown]
# ## Segmentación de los Datos

# %%
features = df.drop("Exited_1", axis=1)
target = df["Exited_1"]

# %%
# Dividiendo los datos
features_train, features_valid_test, target_train, target_valid_test = train_test_split(features, target, test_size=.4, random_state= 12345)
features_valid, features_test, target_valid, target_test = train_test_split(features_valid_test, target_valid_test, test_size=.5, random_state=12345)

# %% [markdown]
# ## Encontrar los mejores hiperparametros

# %%
#Definiendo la metrica principal que nos solicita el cliente
scoring = 'roc_auc'

# %%
# Creando una funcion para elegir los mejores hiperparametros
def find_best_params(models, train_features, train_targets, scoring):
    # Creación de un DataFrame para almacenar los resultados obtenidos
    results = pd.DataFrame(columns=['Model', 'Best Parameters', 'Best Score'])

    # Iteración sobre la lista de modelos
    for model in models:
        # Imprimir el nombre del modelo
        print(f"Encontrando los mejores parametros para {type(model['model']).__name__}...")

        # Utilizando grid search para la busqueda
        grid = GridSearchCV(model['model'], model['param_grid'], cv=5, scoring=scoring, verbose=0, n_jobs=-1)
        grid.fit(train_features, train_targets)

        # Obteniendo los mejores parametros y el score
        best_params = grid.best_params_
        best_score = np.abs(grid.best_score_)

        # Almacenando los resultados en el DataFrame
        results = pd.concat([results, pd.DataFrame({'Model': type(model['model']).__name__, 
                                  'Best Parameters': [best_params], 
                                  'Best Score': best_score})], ignore_index=True)

        # Ordenando los resultados
        results.sort_values(by='Best Score', ascending=False, inplace=True)

    # Creando una grafica para visualizar los resultados
    plt.figure(figsize=(12, 10))
    sns.barplot(data=results, x='Model', y='Best Score', palette="coolwarm")
    plt.title("Grid Search Results")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return results

# %%
print(target_train.dtype)

# %%
label_encoder = LabelEncoder()
target_train_discrete = label_encoder.fit_transform(target_train)

# %%
print(target_train_discrete.dtype)

# %%
target_train =target_train_discrete

# %%
# Define the models and their basic hyperparameters
models = [
    {
        'model': RandomForestClassifier(random_state= 12345, class_weight='balanced'),
        'param_grid': {'n_estimators': np.arange(50, 201, 50), 'max_depth': np.arange(3, 15)}
    },
    {
        'model': DecisionTreeClassifier(random_state=12345, class_weight='balanced'),
        'param_grid': {'max_depth': np.arange(3, 15)}
    },
    {
        'model': GradientBoostingClassifier(random_state=12345),
        'param_grid': {'n_estimators': np.arange(50, 201, 50), 'max_depth': np.arange(3, 15)}
    },
    {
        'model': KNeighborsClassifier(),
        'param_grid': {'n_neighbors': np.arange(1, 11)}
    },
]

# Encontrando los mejores parametros para cada modelo
results = find_best_params(models, features_train, target_train, scoring)

# Imprimir los resultados
results

# %%
# Set the index of the DataFrame to the model name
results.set_index('Model', inplace=True)

# Set the hyperparameters for each model based on the results of the grid search
models = [
    RandomForestClassifier(random_state=12345, **results.loc['RandomForestClassifier']['Best Parameters']),
    DecisionTreeClassifier(random_state=12345, **results.loc['DecisionTreeClassifier']['Best Parameters']),
    GradientBoostingClassifier(random_state=12345, **results.loc['GradientBoostingClassifier']['Best Parameters']),
    KNeighborsClassifier(**results.loc['KNeighborsClassifier']['Best Parameters']),
]

# %% [markdown]
# # Entrenar y evaluar los modelos

# %%
# Create a function to train and evaluate multiple models on the training and validation subsets and create a graph to visualize the results
def train_and_evaluate_models(models, features_train, target_train, features_valid, target_valid):
    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall', 'ROC_AUC'])
    
    # Loop through each model
    for model in models:
        # Train the model
        model.fit(features_train, target_train)
        
        # Make predictions
        predictions = model.predict(features_valid)
        
        # Evaluate the model
        accuracy = model.score(features_valid, target_valid)
        f1 = f1_score(target_valid, predictions)
        precision = precision_score(target_valid, predictions)
        recall = recall_score(target_valid, predictions)
        roc_auc = roc_auc_score(target_valid, predictions)
        
        # Append the results to the DataFrame
        results = pd.concat([results, pd.DataFrame({'Model': model.__class__.__name__, 'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall, 'ROC_AUC': roc_auc}, index=[0])], ignore_index=True)
    
    # Draw a heatmap to visualize the results
    plt.figure(figsize=(14,7))
    sns.set(style="whitegrid")
    sns.heatmap(results.set_index('Model'), annot=True, cmap='RdBu_r')
    plt.title('Model Comparison')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()

    return results

# %%
print(target_valid.dtype)

# %%
label_encoder = LabelEncoder()
target_valid_discrete = label_encoder.fit_transform(target_valid)

# %%
print(target_valid_discrete.dtype)

# %%
target_valid = target_valid_discrete

# %%
# Train and evaluate the models
train_results = train_and_evaluate_models(models, features_train, target_train, features_valid, target_valid)

# Print the results
train_results

# %%
label_encoder = LabelEncoder()
target_test_discrete = label_encoder.fit_transform(target_test)
target_test = target_test_discrete

# %%
# Train and evaluate the models on the test set
test_results = train_and_evaluate_models(models, features_train, target_train, features_test, target_test)

# Print the results
test_results


