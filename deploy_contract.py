from solcx import compile_standard, install_solc
import json
import os
from web3 import Web3
from dotenv import load_dotenv

install_solc("0.8.4")
load_dotenv()

with open("./blacklist.sol", "r") as file:
    blacklist_file = file.read()

### Compile Our Solidity
compiled_sol = compile_standard(
    {
        "language": "Solidity",
        "sources": {"blacklist.sol": {"content": blacklist_file}},
        "settings": {
            "outputSelection": {
                "*": {"*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}
            }
        },
    },
    solc_version="0.8.4",
)
with open("./compiled_code.json", "w") as f:
    json.dump(compiled_sol, f)


### Preparing for Deployment

# Bytecode
bytecode = compiled_sol["contracts"]["blacklist.sol"]["dataDeployment"]["evm"][
    "bytecode"
]["object"]

# ABI
abi = compiled_sol["contracts"]["blacklist.sol"]["dataDeployment"]["abi"]


print("Deploying Ccntract")

# Connecting to ganache
w3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
chain_id = 1337
my_address = "0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1"

private_key = os.getenv("PRIVATE_KEY")

# Create the contract in python
BlackList = w3.eth.contract(abi=abi, bytecode=bytecode)

# Get the latest transaction
nonce = w3.eth.get_transaction_count(my_address)
print(nonce)


# build the transaction
# sign the transaction
# send the transaction

transaction = BlackList.constructor().build_transaction(
    {"chainId": chain_id, "from": my_address, "nonce": nonce}
)
# now signing the transaction using our private key
signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
# print(signed_txn)

# send
tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
# print("Waiting for transaction to finish...")
deployed_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print("Deployed")
deployed_address = deployed_receipt.contractAddress
# saving deployed address
with open(".env", "a+") as en:
    en.write(f'\nexport deployed_address="{deployed_address}"')
    en.write(f"\nexport Bytecode={bytecode}")
    en.write(f"\nexport ABI={abi}")
