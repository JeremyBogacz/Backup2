#include <iostream>
#include "currencyConverter.h"

using namespace std;

int main(int argc, char* argv[]){
	cout << "Showing conversion for Japanese yen." << endl;

	currencyConverter yenConverter(113.59, 2.0);

	cout << "Exchange Rate = " << yenConverter.getExchangeRate() << endl;

	cout << "Exchange Fee = " << yenConverter.getExchangeFee() << endl;

	cout << "100 Japanese yens = " << yenConverter.toDollar(100) << endl;

	cout << "100 US Dollars = " << yenConverter.fromDollar(100) << endl;

	cout << endl;


	cout << "Showing conversion for euro." << endl;

	currencyConverter euroConverter(0.86, 3.5);

	cout << "Exchange Rate = " << euroConverter.getExchangeRate() << endl;

	cout << "Exchange Fee = " << euroConverter.getExchangeFee() << endl;

	cout << "100 euros = " << euroConverter.toDollar(100) << endl;

	cout << "100 US Dollars = " << euroConverter.fromDollar(100) << endl;

	cout << endl;


	currencyConverter currArray[2] = {currencyConverter(0.95, 2.2), currencyConverter(1.3, 2.5)};

	cout << "Exchange Rate for currArray[1] = " << currArray[1].getExchangeRate() << endl;

	cout << "Exchange Fee for currArray[0] = " << currArray[0].getExchangeFee() << endl;

	cout << "100 units of currArray[1] = " << currArray[1].toDollar(100) << " dollars" << endl;

	cout << "100 US dollars = " << currArray[0].fromDollar(100) << " units of currArray[0]" << endl;

	cout << endl;

	for(int i = 0; i < 2; i++){
		cout << "Exchange fee for currArray[" << i << "] = " << currArray[i].getExchangeFee() << endl;
	}

	cout << endl;
	cout << "Done.";
	cout << endl;
}


