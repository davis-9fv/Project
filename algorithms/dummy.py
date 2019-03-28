import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def dummy(x_train, x_val, x_test):
    print('Dummy')
    print("Train VS Val")
    y_val_predicted_list = []
    y_train_val_predicted_list = []
    y_test_predicted_list = []

    y_val_predicted_list.append(x_val[:, -1])

    print("Train + Val VS Train + Val")
    x_train_val = np.concatenate((x_train, x_val), axis=0)
    y_train_val_predicted_list.append(x_train_val[:, -1])

    print("Train + Val VS Test")
    y_test_predicted_list.append(x_test[:, -1])

    return y_val_predicted_list, y_train_val_predicted_list, y_test_predicted_list
