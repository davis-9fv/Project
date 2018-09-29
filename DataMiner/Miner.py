from bs4 import BeautifulSoup as BS
from urllib.request import urlopen
import pandas as pd
from datetime import datetime

# genesis_block = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
genesis_block = "0000000000000000001ecdeebdc3d86b707750aa3c040551fc75a73b7f983d48"
current_block_hash = ""
use_genesis = True

# columns = ['Hash', 'Previous Block', 'Next Block(s)']
columns = ['Hash', 'Previous Block', 'Next Block(s)', 'Number Of Transactions', 'Output Total',
           'Estimated Transaction Volume', 'Transaction Fees', 'Height', 'Timestamp', 'Relayed By', 'Difficulty',
           'Bits', 'Size', 'Weight', 'Version', 'Nonce', 'Block Reward']

df = pd.DataFrame(columns=columns)
print(str(datetime.now()))

for i in range(0, 1):
    if use_genesis:
        current_block_hash = genesis_block
        use_genesis = False

    url = "https://www.blockchain.com/btc/block/" + current_block_hash
    usock = urlopen(url)
    data = usock.read()
    usock.close()
    soup = BS(data)
    # print(soup.find('table',{'class':'table table-striped'}))
    # print(soup.find('div',{'class':'col-md-6 col-sm-6'}))
    # print(soup.find_all('td'))
    # print(soup.get_text())
    # for link in soup.find_all('td'):
    # print(link.get_text())

    print(soup.title.get_text())
    hash = ""
    previous_block = ""
    next_block = ""
    num_transactions = ""
    output_total = ""
    estimated_transaction_value = ""
    transaction_fees = ""
    height = ""
    timestamp = ""
    relayed_by = ""
    difficulty = ""
    bits = ""
    size = ""
    weight = ""
    version = ""
    nonce = ""
    block_reward = ""

    td_html = soup.find_all('td')

    for x in range(len(td_html)):
        link = td_html[x]
        print(link.get_text())
        if link.get_text() == 'Hash':
            hash = td_html[x + 1].get_text()
            hash = hash.rstrip()

        if link.get_text() == 'Previous Block':
            previous_block = td_html[x + 1].get_text()

        if link.get_text() == 'Next Block(s)':
            next_block = td_html[x + 1].get_text()
            next_block = next_block.rstrip()

        if link.get_text() == 'Number Of Transactions':
            num_transactions = td_html[x + 1].get_text()

        if link.get_text() == 'Output Total':
            output_total = td_html[x + 1].get_text()

        if link.get_text() == 'Estimated Transaction Volume':
            estimated_transaction_value = td_html[x + 1].get_text()

        if link.get_text() == 'Transaction Fees':
            transaction_fees = td_html[x + 1].get_text()

        if link.get_text() == 'Height':
            height = td_html[x + 1].get_text()
            height = height.rstrip()

        if link.get_text() == 'Timestamp':
            timestamp = td_html[x + 1].get_text()

        if link.get_text() == 'Relayed By':
            relayed_by = td_html[x + 1].get_text()

        if link.get_text() == 'Difficulty':
            difficulty = td_html[x + 1].get_text()

        if link.get_text() == 'Bits':
            bits = td_html[x + 1].get_text()

        if link.get_text() == 'Size':
            size = td_html[x + 1].get_text()

        if link.get_text() == 'Weight':
            weight = td_html[x + 1].get_text()

        if link.get_text() == 'Version':
            Version = td_html[x + 1].get_text()

        if link.get_text() == 'Nonce':
            nonce = td_html[x + 1].get_text()

        if link.get_text() == 'Block Reward':
            block_reward = td_html[x + 1].get_text()

    # print('----')
    print(hash)
    # print(previous_block)
    # print(next_block)

    # Creation of pandas data frame with columns.
    # columns = [
    #    'Number of Transactions', 'Output Total', 'Estimated Transaction Volume', 'Transaction Fees', 'Timestamp', 'Size',
    #    'Block Reward', 'Hash']

    row = [[hash, previous_block, next_block, num_transactions, output_total, estimated_transaction_value,
            transaction_fees, height, timestamp, relayed_by, difficulty, bits, size, weight, version, nonce,
            block_reward]]
    df2 = pd.DataFrame(row, columns=columns)
    df = df.append(df2)

    df2.to_csv('my_csv.csv', mode='a', header=False)

    current_block_hash = previous_block

print(df)

print(str(datetime.now()))
