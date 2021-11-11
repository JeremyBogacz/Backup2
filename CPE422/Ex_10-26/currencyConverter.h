/** File name: currencyConverter.h
 * *   Class for performing currency conversion between
 * *   a foreign currency and the US dollar.
 * *   Written by Amer Qouneh, October 23, 2021.
 * */

#ifndef CURRENCYCONVERTER_H
#define CURRENCYCONVERTER_H

class currencyConverter {
	private:
		double exchangeRate;
		double exchangeFee;
	public:
		currencyConverter(double rate, double fee);
		void setExchangeRate(double rate);
		double getExchangeRate();
		void setExchangeFee(double fee);
		double getExchangeFee();
		double fromDollar(double dollar);
		double toDollar(double foreignMoney);
		~currencyConverter();
};

#endif
