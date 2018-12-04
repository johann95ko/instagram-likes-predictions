from keras import Sequential, layers
from keras.layers import Dense, Activation
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('./thelanativedatabase.csv', index_col=0)

target = 'likes_followers_ratio'

X = train_df.drop(target, axis=1)
Y = train_df[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(19,)))
model.add(layers.Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(4096, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['mae'])

model.fit(
    X_train,
    Y_train,
    batch_size=100,
    epochs=10,
    shuffle=True,
    verbose=2
)

print(r2_score(model.predict(X_test), Y_test))
print(list(model.predict(X_test)))
print(list(Y_test))