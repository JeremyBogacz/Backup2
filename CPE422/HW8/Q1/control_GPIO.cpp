// Created by Jeremy Bogacz on 11/7/2021
//
// This program takes in two gpio numbers and a state.
// It will set the state of the first gpio number then
// get the state of the second one and print it out.
// The gpios need to be configured in the build file.
// By default gpio60 is configured for output and gpio46
// is configured for input+

#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
using namespace std;

#define PATH "/sys/class/gpio"	// Define path to gpio directory

int main(int argc, char* argv[]){
	// Check to make sure all arguments are entered
	if(argc != 4){
		cout << "Usage is: control_GPIO gpio_number state gpio_number" << endl;
		cout << "gpio_number: GPIO number to control" << endl;
		cout << "state: hi or low" << endl;
		return 2;
	}

	cout << "Starting control_GPIO program:" << endl << endl;
	
	// Read in all three arguments
	int gpioNum1 = atoi(argv[1]);
	string state(argv[2]);
	int gpioNum2 = atoi(argv[3]);
	
	// Set valOut for either high or low state
	string valOut;

	if(state == "hi"){
		valOut = "1";
	}
	else if(state == "low"){
		valOut = "0";
	}
	 // Create full file paths and print them out
	ostringstream s1;
	s1 << PATH << "/gpio" << gpioNum1 << "/value";
	string filePath1 = string(s1.str());

	ostringstream s2;
	s2 << PATH << "/gpio" << gpioNum2 << "/value";
	string filePath2 = string(s2.str());
	
	cout << "File Path 1: " << filePath1 << endl;
	cout << "File Path 2: " << filePath2 << endl << endl;
	
	// Open first gpio value file and write the state
	cout << "Setting GPIO" << gpioNum1 << " state to " << state << endl;

	ofstream fs1;
	fs1.open(filePath1.c_str());
	fs1 << valOut;
	fs1.close();
	
	// Open second gpio value file and read it into valIn, print it out
	cout << "Reading GPIO" << gpioNum2 << endl;
	ifstream fs2;
	fs2.open(filePath2.c_str());
	string valIn;
	getline(fs2,valIn);
        fs2.close();
	
	cout << endl << "The value of GPIO" << gpioNum2 << " is " << valIn << "."  << endl << endl;

	cout << "End of control_GPIO program." << endl;	

}


