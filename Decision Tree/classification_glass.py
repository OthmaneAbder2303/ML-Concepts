import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Chargement et préparation (Prétraitement)
data = pd.read_csv('Data/glass.csv')
X = data.drop('Type', axis=1)
y = data['Type']

# Tâche 3 : Normalisation et séparation 70/30 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Tâche 4 : Construction et entraînement
model = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=5, 
    min_samples_leaf=5
)
model.fit(X_train, y_train)

# Visualisation
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, rounded=True)
plt.show()

# Tâche 5 : Évaluation 
y_pred = model.predict(X_test)
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))


# # Définition de la grille de paramètres à tester
# param_grid = {
#     'max_depth': [3, 5, 10, 15, None],
#     'min_samples_leaf': [1, 2, 5, 10]
# }

# # Initialisation du modèle de base
# tree_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# # Configuration de la recherche par grille
# # cv=5 signifie que l'on fait une validation croisée sur 5 plis
# grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5)

# # Entraînement sur les données d'apprentissage
# grid_search.fit(X_train, y_train)

# # Affichage des meilleurs paramètres trouvés
# print(f"Meilleurs paramètres : {grid_search.best_params_}")

# # Évaluation du meilleur modèle sur l'ensemble de test
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# print("\nNouveau rapport de classification :")
# print(classification_report(y_test, y_pred))