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
#print('score is', linreg.score(X_val,Y_val))

user_input = pd.read_csv('./dataxuserinput.csv')
max_hour = 0
max_likes = 0

#Check across 24hrs
for i in range(24):
    user_input['hour'] = i
    # print(user_input)
    predicted_likes = linreg.predict(user_input)
    # print('hour is', i, 'and predicted likes is', predicted_likes)

    if predicted_likes > max_likes:
        max_likes = predicted_likes
        max_hour = i
print('optimal hour is',max_hour,'and predicted likes is',max_likes)
