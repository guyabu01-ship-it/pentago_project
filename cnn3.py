import keras
from keras import layers, callbacks
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
print("GPU זמין:" if gpus else "רץ על CPU בלבד")

def load_data(filename='processed_data_np.npz'):

    data = np.load(filename)

    X = data['X'].astype('float32')
    y = data['Y'].astype('float32')

    return X, y


X, y = load_data()

# reshape ללמידת CNN
X = X.reshape(-1, 6, 6, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)


# CNN 

model = keras.Sequential([

    layers.Input(shape=(6,6,1)),

    layers.Conv2D(64,(3,3),padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),

    layers.Conv2D(64,(3,3),padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),

    layers.Conv2D(128,(3,3),padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),

    layers.Flatten(),

    layers.Dense(256),
    layers.LeakyReLU(),
    layers.Dropout(0.3),

    layers.Dense(128),
    layers.LeakyReLU(),

    layers.Dense(1,activation='tanh')
])


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae','mse']
)


early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=1024,
    validation_split=0.25,
    callbacks=[early_stop]
)


# graphs 

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mse'], label='Train MSE')
plt.plot(history.history['val_mse'], label='Val MSE')
plt.title("Model MSE")
plt.legend()

plt.savefig('loss_and_mse_comparison_cnn3.png', dpi=300) 

plt.figure(figsize=(6, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("Model MAE")
plt.legend()

plt.savefig('model_mae_resultsc_cnn3.png', dpi=300)

# predict 

def predict_board(board):

    board=np.array(board).reshape(1,6,6,1)

    return model.predict(board)


prediction = predict_board(X_test[0])

print("Prediction:",prediction)



model.save("pentago_cnn_model.keras")
