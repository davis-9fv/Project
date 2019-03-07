from sklearn import neural_network
import numpy as np


def mlpregressor(x_train, y_train, x_val, y_val, x_test):
    alphas = np.linspace(3, 0, 50)
    print('Lasso')
    print(alphas)
    print("Total Alphas %i" % (len(alphas)))

    print("Train VS Val")
    # For time we choose max_iter=100000

    nn = neural_network.MLPRegressor(activation='relu', solver='adam',
                                     batch_size='auto',
                                     hidden_layer_sizes=(10,),
                                     max_iter=1000000000, shuffle=True, early_stopping=True)
    print(nn)
    y_val_predicted_list = []
    y_train_val_predicted_list = []
    y_test_predicted_list = []

    for a in alphas:
        nn.set_params(alpha=a)
        print(a)
        nn.fit(x_train, y_train)
        y_val_predicted_list.append(nn.predict(x_val))

    print("Train + Val VS Train + Val")
    x_train_val = np.concatenate((x_train, x_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    for a in alphas:
        nn.set_params(alpha=a)
        nn.fit(x_train_val, y_train_val)
        y_train_val_predicted_list.append(nn.predict(x_train_val))
        # Train + Val VS Test
        y_test_predicted_list.append(nn.predict(x_test))

    return y_val_predicted_list, y_train_val_predicted_list, y_test_predicted_list
