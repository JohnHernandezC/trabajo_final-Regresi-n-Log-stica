import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# Cargar los datos
data = pd.read_csv("Titanic-Dataset.csv")

# Visualizar las primeras filas del dataset
print('HEADER DEL DATASET')
print(data.head())

# Obtener información general sobre los datos que el dataset nos da 
print('INFORMACION DATOS')
print(data.info())

# Aqui generamos las estadisticas descriptivas 
print('ESTADISTICAS DESCRIPTIVAS')
print(data.describe())

# Visualización de relaciones entre variables
sns.pairplot(data, hue="Survived", palette="husl")
plt.show()



# Tratamos valores faltantes
data.isnull().sum()

# Imputar valores faltantes
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Transformar variables categóricas a numéricas
data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)



X = data.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1)
y = data["Survived"]

best_features = SelectKBest(score_func=chi2, k="all")
fit = best_features.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
feature_scores.columns = ["Feature", "Score"]
print('SCORE')
print(feature_scores.nlargest(8, "Score"))

# SEPARAMOS EL DATASET DE TRAIN Y TEST


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 5: Entrenar el modelo


logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Paso 6: Evaluar el modelo


y_pred = logistic_model.predict(X_test)

print("EXACTITUD:", accuracy_score(y_test, y_pred))
print("PRECISION:", precision_score(y_test, y_pred))
print("RECALL:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nMATRIS DE CONFUSION:\n", confusion_matrix(y_test, y_pred))
print("\nREPORTE DE CLASIFICACION:\n", classification_report(y_test, y_pred))

# Paso 7: Visualizar resultados
# Matriz de confusión
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Paso 8: Interpretar resultados
# Basado en las métricas de evaluación y la matriz de confusión, se puede interpretar la efectividad del modelo en predecir la supervivencia de los pasajeros del Titanic.
