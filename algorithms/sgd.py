from sklearn import linear_model
import numpy as np


def sgd(x_train, y_train, x_val, y_val, x_test):
    alphas = np.linspace(5, 0, 50)
    print('SGD')
    print(alphas)
    print("Total Alphas %i" % (len(alphas)))

    print("Train VS Val")
    # For time we choose max_iter=8000
    sgd = linear_model.SGDRegressor(max_iter=8000, verbose=False, shuffle=False)
    y_val_predicted_list = []
    y_train_val_predicted_list = []
    y_test_predicted_list = []

    for a in alphas:
        sgd.set_params(alpha=a)
        #print(a)
        sgd.fit(x_train, y_train)
        y_val_predicted_list.append(sgd.predict(x_val))

    print("Train + Val VS Train + Val")
    x_train_val = np.concatenate((x_train, x_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    for a in alphas:
        sgd.set_params(alpha=a)
        sgd.fit(x_train_val, y_train_val)
        y_train_val_predicted_list.append(sgd.predict(x_train_val))
        # Train + Val VS Test
        y_test_predicted_list.append(sgd.predict(x_test))

    return y_val_predicted_list, y_train_val_predicted_list, y_test_predicted_list
