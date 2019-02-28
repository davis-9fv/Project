from sklearn import linear_model
import numpy as np


def elasticnet(x_train, y_train, x_val, y_val, x_test):
    alphas = np.linspace(3, 0, 50)
    print('Elasticnet')
    print(alphas)
    print("Total Alphas %i" % (len(alphas)))

    print("Train VS Val")
    elasticnet = linear_model.ElasticNet(max_iter=100000, normalize=False)
    y_val_predicted_list = []
    y_train_val_predicted_list = []
    y_test_predicted_list = []

    for a in alphas:
        elasticnet.set_params(alpha=a)
        print(a)
        elasticnet.fit(x_train, y_train)
        y_val_predicted_list.append(elasticnet.predict(x_val))

    print("Train + Val VS Train + Val")
    x_train_val = np.concatenate((x_train, x_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    for a in alphas:
        elasticnet.set_params(alpha=a)
        elasticnet.fit(x_train_val, y_train_val)
        y_train_val_predicted_list.append(elasticnet.predict(x_train_val))
        # Train + Val VS Test
        y_test_predicted_list.append(elasticnet.predict(x_test))

    return y_val_predicted_list, y_train_val_predicted_list, y_test_predicted_list
