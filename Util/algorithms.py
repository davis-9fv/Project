from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import linear_model
from keras.layers import Dropout
from keras.regularizers import L1L2


def elastic_net(x_train, y_train, x_to_predict, normalize=False):
    # Alpha optimazed 0.0612244898
    regr = ElasticNet(random_state=0, max_iter=1800000, alpha=0.0612244898, normalize=normalize)
    regr.fit(x_train, y_train)
    y_predicted = regr.predict(x_to_predict)
    return y_predicted, regr


def lasso(x_train, y_train, x_to_predict, normalize=False):
    # Alpha optimazed 1.40816326531
    regr = linear_model.Lasso(max_iter=1800000, alpha=1.40816326531, normalize=normalize)
    regr.fit(x_train, y_train)
    y_predicted = regr.predict(x_to_predict)
    return y_predicted, regr


def knn_regressor(x_train, y_train, x_to_predict, n_neighbors=5):
    regr = KNeighborsRegressor(algorithm='kd_tree', leaf_size=30, weights='uniform', n_neighbors=n_neighbors, n_jobs=4)
    regr.fit(x_train, y_train)
    y_predicted = regr.predict(x_to_predict)
    return y_predicted, regr


def sgd_regressor(x_train, y_train, x_to_predict):
    # Alpha optimazed 0.00000
    regr = linear_model.SGDRegressor(max_iter=2000, alpha=0.00000, verbose=False, shuffle=False)
    regr.fit(x_train, y_train)
    y_predicted = regr.predict(x_to_predict)
    return y_predicted, regr


# fit an LSTM network to training data
# To tune https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# nb_epoch: 10, 100, 500, 1000
# batch_size: 32, 64, 128
def lstm(x_train, y_train, x_to_predict, batch_size, nb_epoch=3, neurons=3):
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    model = Sequential()

    model.add(LSTM(neurons,
                   #bias_regularizer=L1L2(l1=0.01, l2=0.01),
                   batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),
                   stateful=True))

    # model.add(Dense(6, activation='relu'))
    # model.add(Dense(1,activation='linear'))
    # model.add(Dropout(0.001))
    model.add(Dense(1))
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

    return y_predicted, model


def lstm_predict(model, x_to_predict, batch_size):
    y_predicted = list()
    for i in range(len(x_to_predict)):
        X = x_to_predict[i]
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        y_predicted.append(yhat[0, 0])

    return y_predicted
