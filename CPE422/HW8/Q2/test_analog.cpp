// Created by Jeremy Bogacz on 11/9/2021
//
// This program tests the analogIn class.
// It created an analogIn object initialized
// to zero and then get the number and reads 
// the adc value.

#include "analogIn.h"

int main(int argc, char* argv[]){
	cout << "Starting test_analog program." << endl << endl;

	analogIn AIN0(0);	// Create analogIn with ADC value 0

	int adcNum = AIN0.getNumber();	// Get ADC number
	int adcVal = AIN0.readADCsample();	// Get ADC sample value

	cout << "Selected ADC channel: " << adcNum << endl;
	cout << "Sampled ADC value: " << adcVal << endl << endl;

	cout << "End of program." << endl;

	return 0;
}
