from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import linear_model


def elastic_net(x_train, y_train, x_to_predict, y_test, normalize=False):
    # max_iter = 12000, alpha = 0.1
    regr = ElasticNet(random_state=0, max_iter=1800000, alpha=0.0000001, normalize=normalize)
    regr.fit(x_train, y_train)
    # print('X_to_predict:')
    # print(x_to_predict)

    y_predicted = list()
    columns = x_to_predict.shape[1]
    for i in range(len(x_to_predict)):
        x = x_to_predict[i]
        y_hat = float(regr.predict([x]))
        y_predicted.append(y_hat)

    row = list()
    for i in range(len(x_to_predict)):
        row = list()
        row.append(y_test[i])
        for j in range(columns - 1):
            row.append(x_to_predict[i][j])
        # print(row)

    # print('predicted')
    x = x_to_predict[0]
    x_future = list()
    y_future = list()
    """""
    for i in range(len(x_to_predict)):
        print([x])
        y_hat = regr.predict([x])
        y_hat = float(y_hat)
        y_future.append((float(y_hat)))
        row = []
        row.append(y_hat)

        for j in range(columns - 1):
            row.append(x[j])
        #print(row)
        x_future.append(row)
        x = row
    """
    # y_predicted
    # print('y_predicted')
    # print(y_predicted)
    # print('y_future')
    # print(y_future)

    return y_predicted, y_future


def elastic_net2(x_train, y_train, x_to_predict,normalize=False):
    regr = ElasticNet(random_state=0, max_iter=1800000,alpha=0.0000001, normalize=normalize)
    regr.fit(x_train, y_train)
    y_predicted = regr.predict(x_to_predict)
    return y_predicted


def knn_regressor(x_train, y_train, x_to_predict, n_neighbors=5):
    neigh = KNeighborsRegressor(algorithm='kd_tree', leaf_size=30, weights='uniform', n_neighbors=n_neighbors, n_jobs=4)
    neigh.fit(x_train, y_train)
    y_predicted = neigh.predict(x_to_predict)
    return y_predicted


def lasso(x_train, y_train, x_to_predict, normalize=False):
    clf = linear_model.Lasso(max_iter=1800000, alpha=0.00005, normalize=normalize)
    clf.fit(x_train, y_train)
    return clf.predict(x_to_predict)


def sgd_regressor(x_train, y_train, x_to_predict):
    clf = linear_model.SGDRegressor(max_iter=2000, verbose=False, shuffle=False)
    clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_to_predict)
    return y_predicted


# fit an LSTM network to training data
# To tune https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# nb_epoch: 10, 100, 500, 1000
# batch_size: 32, 64, 128
def lstm(x_train, y_train, x_to_predict, batch_size, nb_epoch=3, neurons=3):
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    model = Sequential()

    model.add(LSTM(neurons, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]), stateful=True))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1,activation='linear'))
    #model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    for i in range(nb_epoch):
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

    y_predicted = list()
    for i in range(len(x_to_predict)):
        X = x_to_predict[i]
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        y_predicted.append(yhat[0, 0])

    return y_predicted


def dummy(data):
    return "as"
