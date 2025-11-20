import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ======================================================
#   1) Préparation des labels t
# ======================================================

t = np.zeros((np.shape(data)[0],))
t[:N1] = 1      # Les N1 premiers points -> classe 1

# ======================================================
#   2) Régression linéaire
# ======================================================

MyRegression = LinearRegression()
MyRegression.fit(data, t)

# ======================================================
#   3) Définition de la grille d’affichage
# ======================================================

x1min = np.min(data[:, 0])
x1max = np.max(data[:, 0])
x2min = np.min(data[:, 1])
x2max = np.max(data[:, 1])

x1 = np.linspace(x1min, x1max, 100)
x2 = np.linspace(x2min, x2max, 100)

xx1, xx2 = np.meshgrid(x1, x2)

# Feature matrix pour chaque point de la grille
Xgrid = np.vstack((xx1.flatten(), xx2.flatten())).T

# ======================================================
#   4) Prédiction sur la grille
# ======================================================

prediction = MyRegression.predict(Xgrid)

# seuil de classification
prediction = prediction > 0.5

# ======================================================
#   5) Visualisation
# ======================================================

cm_bright = ListedColormap(["#FF0000", "#0000FF"])

plt.contourf(
    xx1, xx2,
    prediction.reshape(np.shape(xx1)),
    alpha=0.2,
    cmap=cm_bright
)

plt.scatter(data[:, 0], data[:, 1], c=t, cmap=cm_bright)

plt.show()
