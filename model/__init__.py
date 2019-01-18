import numpy as np

print(np.log(4))


def get_value(diff, percentage, start):
    ans = (diff * percentage / 100) + start
    print(ans)
    return ans


data = [5, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 14]
new_data = data
print(data[0:101])

percentage_1 = 14.29
percentage_2 = 28.58
percentage_3 = 42.87
percentage_4 = 57.16
percentage_5 = 71.45
percentage_6 = 85.74

# sub_data = data[0:8]
# sub_data = data[7:15]
start_block = 0
end_block = 8

while len(data) >= end_block:

    sub_data = data[start_block:end_block]
    print(sub_data)
    start = sub_data[0]
    end = sub_data[len(sub_data) - 1]
    diff = end - start
    value_1 = get_value(diff, percentage_1, start)
    value_2 = get_value(diff, percentage_2, start)
    value_3 = get_value(diff, percentage_3, start)
    value_4 = get_value(diff, percentage_4, start)
    value_5 = get_value(diff, percentage_5, start)
    value_6 = get_value(diff, percentage_6, start)

    new_data[start_block+1] = value_1
    new_data[start_block+2] = value_2
    new_data[start_block+3] = value_3
    new_data[start_block+4] = value_4
    new_data[start_block+5] = value_5
    new_data[start_block+6] = value_6

    start_block = end_block - 1
    end_block = start_block + 8


print(new_data)