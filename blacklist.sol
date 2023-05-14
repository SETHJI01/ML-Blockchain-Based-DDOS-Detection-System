// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

contract dataDeployment {

    mapping (string => string) Users;
    string[] public blackListIPAddress;

    function addData(string memory IPAddress,string memory TimeStamp) public{
        Users[IPAddress] = TimeStamp;
        blackListIPAddress.push(IPAddress);
    }

    //Returns Empty String is case of the IP Address being absent
    function checkData(string memory IPAddress) public view returns (string memory) {
        return Users[IPAddress];
    }

    //To delete the IP address from blockchain
    function deleteData(string memory IPAddress) public{
        Users[IPAddress]="";
    }

    //To get all the blacklisted Ip addresses till now
    function getIPs() view public returns (string[] memory){
        return blackListIPAddress;
    }

    //To get count of all the blacklisted IP address till now
    function countIPs() view public returns (uint) {
        return blackListIPAddress.length;
    }

}