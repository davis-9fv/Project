from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model


def elastic_net(x_train, y_train, x_to_predict):
    regr = ElasticNet(random_state=0)
    regr.fit(x_train, y_train)
    y_predicted = regr.predict(x_to_predict)
    return y_predicted


def knn_regressor(x_train, y_train, x_to_predict, n_neighbors=5):
    neigh = KNeighborsRegressor(algorithm='kd_tree', leaf_size=30, weights='uniform', n_neighbors=n_neighbors, n_jobs=4)
    neigh.fit(x_train, y_train)
    y_predicted = neigh.predict(x_to_predict)
    return y_predicted


def sgd_regressor(x_train, y_train, x_to_predict):
    clf = linear_model.SGDRegressor(max_iter=2000, verbose=False, shuffle=False)
    clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_to_predict)
    return y_predicted
