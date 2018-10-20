import pandas as pd
from pandas import read_csv

path = 'C:/tmp/bitcoin/'


def count_rows(df_column):
    print(df_column.shape)


write_file = True

# height_col_name = 'height'
# hash_col_name = 'hash'
# time_col_name = 'time'
# miner_col_name = 'miner'

block_height_file = 'block_height'
block_hash_file = 'block_hash'
block_time_file = 'block_time'
block_miner_file = 'block_miner'
block_transaction_count_file = 'block_transaction_count'
block_witness_tx_count = 'block_witness_count'
block_input_count_file = 'block_input_count'
block_output_count_file = 'block_output_count'
block_input_total_btc_file = 'block_input_total_btc'
block_input_total_usd_file = 'block_input_total_usd'
block_output_total_btc_file = 'block_output_total_btc'
block_output_total_usd_file = 'block_output_total_usd'
block_fee_total_btc_file = 'block_fee_total_btc'
block_fee_total_usd_file = 'block_fee_total_usd'
block_generation_file = 'block_generation'
block_reward_file = 'block_reward'
block_size_file = 'block_size'
block_weight_file = 'block_weight'
block_stripped_size_file = 'block_stripped_size'
block_difficulty_file = 'block_difficulty'

# columns = [height_col_name, hash_col_name, time_col_name, miner_col_name]

df_height = read_csv(path + block_height_file, header=0, error_bad_lines=False)
df_hash = read_csv(path + block_hash_file, header=0, error_bad_lines=False)
df_time = read_csv(path + block_time_file, header=0, error_bad_lines=False)
df_miner = read_csv(path + block_miner_file, header=0, error_bad_lines=False)
df_transaction_count = read_csv(path + block_transaction_count_file, header=0, error_bad_lines=False)
df_witness_tx_count = read_csv(path + block_witness_tx_count, header=0, error_bad_lines=False)
df_input_count = read_csv(path + block_input_count_file, header=0, error_bad_lines=False)
df_output_count = read_csv(path + block_output_count_file, header=0, error_bad_lines=False)
df_input_total_btc = read_csv(path + block_input_total_btc_file, header=0, error_bad_lines=False)
df_input_total_btc.loc[:, 'input_total'] /= 100000000
df_input_total_usd = read_csv(path + block_input_total_usd_file, header=0, error_bad_lines=False)
df_input_total_usd.loc[:, 'input_total_usd'] /= 10000

df_output_total_btc = read_csv(path + block_output_total_btc_file, header=0, error_bad_lines=False)
df_output_total_btc.loc[:, 'output_total'] /= 100000000
df_output_total_usd = read_csv(path + block_output_total_usd_file, header=0, error_bad_lines=False)
df_output_total_usd.loc[:, 'output_total_usd'] /= 10000

df_fee_total_btc = read_csv(path + block_fee_total_btc_file, header=0, error_bad_lines=False)
df_fee_total_btc.loc[:, 'fee_total'] /= 100000000


df_fee_total_usd = read_csv(path + block_fee_total_usd_file, header=0, error_bad_lines=False)
df_fee_total_usd.loc[:, 'fee_total_usd'] /= 10000

df_generation = read_csv(path + block_generation_file, header=0, error_bad_lines=False)
df_generation.loc[:, 'generation'] /= 100000000

df_reward = read_csv(path + block_reward_file, header=0, error_bad_lines=False)
df_reward.loc[:, 'reward'] /= 100000000

df_size_count = read_csv(path + block_size_file, header=0, error_bad_lines=False)
df_weight = read_csv(path + block_weight_file, header=0, error_bad_lines=False)
df_stripped_size = read_csv(path + block_stripped_size_file, header=0, error_bad_lines=False)
df_difficulty = read_csv(path + block_difficulty_file, header=0, error_bad_lines=False)


count_rows(df_height)
count_rows(df_hash)
count_rows(df_time)
count_rows(df_miner)
count_rows(df_transaction_count)
count_rows(df_witness_tx_count)
count_rows(df_input_count)
count_rows(df_output_count)
count_rows(df_input_total_btc)
count_rows(df_input_total_usd)
count_rows(df_output_total_btc)
count_rows(df_output_total_usd)
count_rows(df_fee_total_btc)
count_rows(df_fee_total_usd)
count_rows(df_generation)
count_rows(df_reward)
count_rows(df_size_count)
count_rows(df_weight)
count_rows(df_stripped_size)
count_rows(df_difficulty)

df_result = pd.concat([df_height,
                       df_hash,
                       df_time,
                       df_miner,
                       df_transaction_count,
                       df_witness_tx_count,
                       df_input_count,
                       df_output_count,
                       df_input_total_btc,
                       df_input_total_usd,
                       df_output_total_btc,
                       df_output_total_usd,
                       df_fee_total_btc,
                       df_fee_total_usd,
                       df_generation,
                       df_reward,
                       df_size_count,
                       df_weight,
                       df_stripped_size,
                       df_difficulty], axis=1)

row = 1000

print('Height               ' + str(df_result['id'].iloc[row]))
print('Hash                 ' + df_result['hash'].iloc[row])
print('Mined on             ' + df_result['time'].iloc[row])
print('Miner                ' + df_result['guessed_miner'].iloc[row])
print('df_transaction_count ' + str(df_result['transaction_count'].iloc[row]))
print('df_witness_tx_count  ' + str(df_result['witness_count'].iloc[row]))
print('df_input_count       ' + str(df_result['input_count'].iloc[row]))
print('df_output_count      ' + str(df_result['output_count'].iloc[row]))
print('df_input_total_btc   ' + str(df_result['input_total'].iloc[row]))
print('df_input_total_usd   ' + str(df_result['input_total_usd'].iloc[row]))
print('df_output_total_btc  ' + str(df_result['output_total'].iloc[row]))
print('df_output_total_usd  ' + str(df_result['output_total_usd'].iloc[row]))

print('df_fee_total_btc  ' + str(df_result['fee_total'].iloc[row]))
print('df_fee_total_usd  ' + str(df_result['fee_total_usd'].iloc[row]))
print('df_generation     ' + str(df_result['generation'].iloc[row]))
print('df_reward         ' + str(df_result['reward'].iloc[row]))
print('df_size_count     ' + str(df_result['size'].iloc[row]))
print('df_weight         ' + str(df_result['weight'].iloc[row]))
print('df_stripped_size  ' + str(df_result['stripped_size'].iloc[row]))
print('df_difficulty     ' + str(df_result['difficulty'].iloc[row]))

if write_file:
    df_result.to_csv('C:/tmp/bitcoin/bitcoin.csv', mode='a', header=True)

# print(df_result)
