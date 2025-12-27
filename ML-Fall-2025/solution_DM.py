import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Chargement des données
(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()

# Sélection des classes 0 (avion) et 1 (voiture)
train_mask = (y_train_full[:, 0] == 0) | (y_train_full[:, 0] == 1)
test_mask = (y_test_full[:, 0] == 0) | (y_test_full[:, 0] == 1)

X_train = X_train_full[train_mask]
y_train = y_train_full[train_mask]
X_test = X_test_full[test_mask]
y_test = y_test_full[test_mask]

# Normalisation
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding des labels
y_train = to_categorical(np.where(y_train == 0, 0, 1), num_classes=2)
y_test = to_categorical(np.where(y_test == 0, 0, 1), num_classes=2)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# %%
# Construction du modèle
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

# Applatissement
model.add(Flatten())

# Couches denses
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 neurones pour 2 classes

model.summary()

# %%
# Compilation
from tensorflow.keras.optimizers import Adam

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# %%
# Entraînement
batch_size = 32
epochs = 10

history = model.fit(
    X_train, 
    y_train,  # Déjà en one-hot encoding
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split=0.2
)

# %%
# Évaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")