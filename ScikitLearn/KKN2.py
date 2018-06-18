import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
from sklearn.neighbors import KNeighborsRegressor


raw_data = np.column_stack((X, y))

train, test = raw_data[0:6], raw_data[6:12]


X, y = train[:, 0], train[:, 1]
print(X)
print(X.shape)
X = X.reshape(X.shape[0], 1)
print(X.shape)
neigh = KNeighborsRegressor(4)
neigh.fit(X, y)
predictions = list()

for i in range(len(test)):
    X, y = test[i, 0], test[i, 1]
    yHat = neigh.predict(X)
    predictions.append(yHat)

print(predictions)

flag = True
for i in range(100):
    z = i*4

    if flag:
        flag = False
        print(str(i) + "," + str(z))

    else:
        flag = True

