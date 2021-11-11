#include "bankAccount.h"

int main(){
	// Initialize two bankAccount objects and an array of three objects
	bankAccount jackAccount("Jack Smith", 50.0);
        bankAccount jillAccount("Jill Brian", 100.0);
	bankAccount bankArray[3] = {bankAccount("Ralph Kramden", 10.0), bankAccount("Alice Kramden", 20.0), bankAccount("Ed Norton", 30.0)};
	
	// Deposit $30 into each account
	jackAccount.deposit(30.0);
	jillAccount.deposit(30.0);

	for(int i = 0; i < 3; i++){
		bankArray[i].deposit(30.0);
	}
	
	// Withdraw $20 from each account
	jackAccount.withdraw(20.0);
	jillAccount.withdraw(20.0);

	for(int i = 0; i < 3; i++){
		bankArray[i].withdraw(20.0);
	}
	
	// Print out the names and the balances
	string name;
	float balance;

	cout << endl;

	name = jackAccount.getName();
	balance = jackAccount.getBalance();
	cout << "Account Name: " << name << "   Balance: $" << balance << endl;

	name = jillAccount.getName();
	balance = jillAccount.getBalance();
	cout << "Account Name: " << name << "   Balance: $" << balance << endl;

	for(int i = 0; i < 3; i++){
		name = bankArray[i].getName();
		balance = bankArray[i].getBalance();
		cout << "Account Name: " << name << "   Balance: $" << balance << endl;
	}

	cout << endl;
}
