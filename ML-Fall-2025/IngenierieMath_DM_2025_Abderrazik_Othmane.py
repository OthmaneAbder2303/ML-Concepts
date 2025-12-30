#%pip install tensorflow matplotlib scikit-learn -q
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
# on charge toutes les données
(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
# Sélection que des deux classes  : 0 (avion) et 1 (voiture)
train_mask = (y_train_full[:, 0] == 0) | (y_train_full[:, 0] == 1)
test_mask = (y_test_full[:, 0] == 0) | (y_test_full[:, 0] == 1)
X_train = X_train_full[train_mask]
y_train = y_train_full[train_mask]
X_test = X_test_full[test_mask]
y_test = y_test_full[test_mask]

print(X_train.shape, y_train.shape)
# Normalisation
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
# One-hot encoding des labels
y_train = to_categorical(np.where(y_train == 0, 0, 1), num_classes=2)
y_test = to_categorical(np.where(y_test == 0, 0, 1), num_classes=2)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()

# Bloc 1
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Bloc 2
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Bloc 3
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Applatissement
model.add(Flatten())

# Couches denses
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 neurones pour 2 classes sinon selon le nombre de classes

model.summary()
# Compilation
from tensorflow.keras.optimizers import Adam

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Entraînement
batch_size = 32
epochs = 10

history = model.fit(
    X_train, 
    y_train,
    batch_size=batch_size, 
    epochs=epochs,
    validation_split=0.2
)
# Évaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
# Visualisation
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()
# Confusion Matrix
from sklearn.metrics import confusion_matrix
#%pip install seaborn -q
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)
np.save('IngenierieMath_DM_2025_Abderrazik_Othmane.npy', y_pred)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Avion', 'Voiture'], yticklabels=['Avion', 'Voiture'])
plt.show()
# Je charge le fichier que je viens de créer
data = np.load('IngenierieMath_DM_2025_Abderrazik_Othmane.npy')

print("==> Vérification du fichier .npy")
print(f"Forme des données : {data.shape}")  # c exactement le nombre d'images de test
print(f"Type de données : {data.dtype}")
print(f"5 premières prédictions : {data[:5]}")