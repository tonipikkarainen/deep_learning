import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

tf.disable_v2_behavior()

from ex2 import EXTRA_TRAIN_FILE_PATH

df = pd.read_csv(EXTRA_TRAIN_FILE_PATH)
df_dropped = df.dropna(axis=0, how="any")

df_ind = df_dropped.loc[:, ["make", "body-style", "wheel-base", "engine-size", "horsepower", "peak-rpm", "highway-mpg"]]
df_dep = df_dropped.loc[:, "price"]

# Label encoding
encoder_make = LabelEncoder()
df_ind.loc[:, "make"] = encoder_make.fit_transform(df_ind.loc[:, "make"])

encoder_bodystyle = LabelEncoder()
body_style_pre_encode = np.array(df_ind.loc[:, "body-style"])
df_ind.loc[:, "body-style"] = encoder_bodystyle.fit_transform(df_ind.loc[:, "body-style"])

# Scaler
sc_X = StandardScaler()
X = sc_X.fit_transform(df_ind)

sc_Y = StandardScaler()
Y = sc_Y.fit_transform(df_dep.values.reshape(-1, 1))

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Model
model = Sequential()
model.add(Dense(units=128, activation="relu", input_dim=7))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=1))

model.compile(loss="mean_squared_error", optimizer="adam",
              metrics=["mae"])

# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, verbose=0)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()

# Encode predict value
X_predict = np.array([["audi", "hatchback", 99.5, 131, 160, 5500, 22]])
X_predict[:, 0] = encoder_make.transform(X_predict[:, 0]).reshape(-1, 1)
X_predict[:, 1] = encoder_bodystyle.transform(X_predict[:, 1]).reshape(-1, 1)
print("Encoded predict value: ", X_predict)

# Scaler predict value
X_predict = sc_X.transform(X_predict)

# Keras model
print()
print("Keras")
mse_value, mae = model.evaluate(X_test, Y_test, verbose=0)
print("Evaluation MSE: " + str(mse_value))
print("Evaluation MAE: " + str(mae))

Y_predict = model.predict(X_predict)
print("Predicted value for prediction sample", sc_Y.inverse_transform(Y_predict))

# Scikit learning
print()
print("Scikit learning")
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# Predecting the Result
Y_pred_2 = regressor.predict(X_predict)
print("Predicted value for prediction set: ", sc_Y.inverse_transform(Y_pred_2))

pred_test_lr = regressor.predict(X_test)
print("MSE for test set ", np.sqrt(mean_squared_error(Y_test, pred_test_lr)))