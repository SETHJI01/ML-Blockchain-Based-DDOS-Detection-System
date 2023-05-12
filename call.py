from transactions import addData, checkData

ip_address = "1.2.3.4"
timestamp = "2"
addData(ip_address, timestamp)

print(checkData(ip_address))
