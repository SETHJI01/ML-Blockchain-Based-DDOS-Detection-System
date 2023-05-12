import json
import os
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

deployed_address = os.getenv("deployed_address")
my_address = os.getenv("my_address")
private_key = os.getenv("PRIVATE_KEY")
with open("compiled_code.json") as f:
    compiled_sol = json.load(f)

abi = compiled_sol["contracts"]["blacklist.sol"]["dataDeployment"]["abi"]

w3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
chain_id = 1337
nonce = w3.eth.get_transaction_count(my_address)
print(nonce)
# working with contract
# contract address
# contract ABI
blacklist_storage = w3.eth.contract(address=deployed_address, abi=abi)


# # Call -> Simulate moaking the call and getting a return value
# # Transact -> actually make a state change
# print(blacklist_storage.functions.addData("1.1.1.1","126").call())
def addData(ip_address, timestamp):
    store_transaction = blacklist_storage.functions.addData(
        ip_address, timestamp
    ).build_transaction({"chainId": chain_id, "from": my_address, "nonce": nonce})
    signed_store_txn = w3.eth.account.sign_transaction(store_transaction, private_key)
    send_store_tx = w3.eth.send_raw_transaction(signed_store_txn.rawTransaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(send_store_tx)


def checkData(ip_address):
    return blacklist_storage.functions.checkData(ip_address).call()
