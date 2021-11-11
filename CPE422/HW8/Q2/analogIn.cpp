#include "analogIn.h"

analogIn::analogIn(){
	number = 0;
}

analogIn::analogIn(unsigned int n){
	number = n;
}

void analogIn::setNumber(unsigned int n){
	number = n;
}

int analogIn::readADCsample(){
	ostringstream s;
	s << PATH_AIN << "/in_voltage" << number << "_raw";
	string path = string(s.str());

	ifstream fs;
	fs.open(path.c_str());
	string In;
	getline(fs,In);
	fs.close();

	return stoi(In);	
}

analogIn::~analogIn(){
}
