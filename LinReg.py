import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train_df = pd.read_csv('./thelanativedatabase.csv', index_col=0)

target = 'likes_followers_ratio'

X = train_df.drop(target, axis=1)
X = train_df.drop('comment_count', axis=1)
Y = train_df[target]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15)

linreg = LinearRegression()
linreg.fit(X_train, Y_train)


user_input = pd.read_csv('./dataxuserinput.csv')
print(linreg.predict(user_input))