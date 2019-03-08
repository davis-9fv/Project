import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def knn(x_train, y_train, x_val, y_val, x_test, n_neighbors):
    print('KNN')
    print(n_neighbors)
    print('Total Neighbors %i' % (len(n_neighbors)))

    print("Train VS Val")
    neigh = KNeighborsRegressor(algorithm='kd_tree', leaf_size=30, weights='uniform', n_jobs=-1)
    y_val_predicted_list = []
    y_train_val_predicted_list = []
    y_test_predicted_list = []

    for a in n_neighbors:
        neigh.set_params(n_neighbors=a)
        neigh.fit(x_train, y_train)
        y_val_predicted_list.append(neigh.predict(x_val))

    print("Train + Val VS Train + Val")
    x_train_val = np.concatenate((x_train, x_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    for a in n_neighbors:
        neigh.set_params(n_neighbors=a)
        neigh.fit(x_train_val, y_train_val)
        y_train_val_predicted_list.append(neigh.predict(x_train_val))
        # Train + Val VS Test
        y_test_predicted_list.append(neigh.predict(x_test))

    return y_val_predicted_list, y_train_val_predicted_list, y_test_predicted_list
