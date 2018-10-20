from Util import data_misc

# data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
data = [112, 118, 132, 129, 121, 135, 148, 148,
        136, 119, 104, 118, 115, 126, 141, 135,
        125, 149, 170]
diff_size = 1
diff_values = data_misc.difference(data, diff_size)
diff_values = diff_values.values
size_diff_values = len(diff_values)
split = int(size_diff_values * 0.80)
train, test = diff_values[0:split], diff_values[split:]

predictions = list()
predictions2 = list()
train_tmp = list()
for i in range(len(train)):
    train_tmp.append(train[i])

for i in range(len(test)):
    yhat = test[i]
    yhat = data_misc.inverse_difference(data, yhat, len(test) + 1 - i)
    a = data_misc.inverse_difference2(data[0], train_tmp, test[i])
    train_tmp.append(test[i])
    predictions.append(yhat)
    predictions2.append(a)

print(predictions)
print(predictions2)
print('end')


