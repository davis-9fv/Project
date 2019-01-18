from pandas import unique


def cat_to_num(data):
    categories = unique(data)
    features = []
    for cat in categories:
        binary = (data == cat)
        features.append(binary.astype("int"))
    return features


data = [[1, 1, 1], [2, 1, 2], [1, 2, 3]]
data2 = [2, 1, 1,3,3]
new_data = cat_to_num(data2)
print(new_data)

print(new_data[0])
