#include "derek_LED.h"
int main(int argc, char* argv[]){
	if(argc!=4){
		cout << "Usage is myBlink <command>" << endl;
		cout << "   command is one of: on, off, flash, status or blink" << endl;
		cout << " e.g. myBlink blink 0 5 " << endl;
		cout << "      myBlink on 1 0 " << endl;
		return 2;
	}

	cout << "Starting the myBlink program" << endl;
	
	string cmd(argv[1]);
	int nLED = atoi(argv[2]);
	int n = atoi(argv[3]);
	
	LED leds[4] = { LED(0), LED(1), LED(2), LED(3) };
	
	if(cmd=="on")leds[nLED].turnOn();
	else if(cmd=="off")leds[nLED].turnOff();
	else if(cmd=="flash")leds[nLED].flash("100"); //default is "50"
	else if(cmd=="status")leds[nLED].outputState();
	else if(cmd=="blink")leds[nLED].blink(n);
	else{cout << "Invalid command!" << endl;} 
      
	cout << "Finished the myBlink program" << endl;
	return 0;
}

