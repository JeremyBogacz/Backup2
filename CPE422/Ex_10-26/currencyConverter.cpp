/* File name: currencyConverter.cpp
 *
 * Class for performing currency conversion between
 * a foreign currency and the US dollar.
 * Written by Amer Qouneh, October 23, 2021.
 */


#include <iostream>
#include "currencyConverter.h"

using namespace std;

currencyConverter::currencyConverter(double rate, double fee){
	exchangeRate = rate;
	exchangeFee = fee;
}

void currencyConverter::setExchangeRate(double rate){
	exchangeRate = rate;
}

double currencyConverter::getExchangeRate(){
	return exchangeRate;
}

void currencyConverter::setExchangeFee(double fee){
	exchangeFee = fee;
}

double currencyConverter::getExchangeFee(){
	return exchangeFee;
}

double currencyConverter::fromDollar(double dollar){
	return dollar * exchangeRate;
}

double currencyConverter::toDollar(double foreignMoney){
	return foreignMoney / exchangeRate;
}

currencyConverter::~currencyConverter(){
	cout << "Destroying the conversion objects before exiting." << endl;
}

