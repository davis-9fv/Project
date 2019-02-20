from sklearn import linear_model
import numpy as np


def lasso(x_train, y_train, x_val, y_val, x_test):
    alphas = np.linspace(3, -3, 50)
    print('Lasso')
    print(alphas)
    print("Total Alphas %i" % (len(alphas)))

    print("Train VS Val")
    lasso = linear_model.Lasso(max_iter=1000000, normalize=False)
    y_val_predicted_list = []
    y_train_val_predicted_list = []
    y_test_predicted_list = []

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(x_train, y_train)
        y_val_predicted_list.append(lasso.predict(x_val))

    print("Train + Val VS Train + Val")
    x_train_val = np.concatenate((x_train, x_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(x_train_val, y_train_val)
        y_train_val_predicted_list.append(lasso.predict(x_train_val))
        # Train + Val VS Test
        y_test_predicted_list.append(lasso.predict(x_test))

    return y_val_predicted_list, y_train_val_predicted_list, y_test_predicted_list
