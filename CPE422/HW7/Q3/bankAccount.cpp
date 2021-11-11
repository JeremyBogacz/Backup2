#include "bankAccount.h"

void bankAccount::setName(string n){
	customerName = n;
}

void bankAccount::setBalance(float bal){
	balance = bal;
}

bankAccount::bankAccount(string n, float bal){
	setName(n);
	setBalance(bal);
}

string bankAccount::getName(){
	return customerName;
}

float bankAccount::getBalance(){
	return balance;
}

void bankAccount::deposit(float dep){
	balance = balance +  dep;

	cout << endl <<  "Depositing $" << dep << " into " << customerName << "'s account." << endl;
}

void bankAccount::withdraw(float draw){
	balance = balance - draw;

	cout << endl <<  "Withdrawing $" << draw << " from " << customerName << "'s account." << endl;
}
