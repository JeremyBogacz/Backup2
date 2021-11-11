/** A program that opens a text file, reads and prints line by line to the screen.  
 * *   The text file name is passed on the command line as an argument.
 * *   The text file should be placed in /home/debian for this to work.
 * */

#include<iostream>
#include<fstream>
#include<string>
using namespace std;

#define FILE_PATH "/home/debian/CPE422"


int main(int argc, char* argv[]){
  if(argc!=2){
	  cout << endl;
	  cout << "Usage is reading-file and name of file: " << endl;
	  cout << " e.g. reading-file sample.txt" << endl;
	  return 2;
  }
  string cmd(argv[1]);
  cout << endl;
  cout << "Starting the reading-file program" << endl;
  cout << "The current FILE Path is: " << FILE_PATH << endl;
  cout << "File name passed is: " << cmd << endl;
  cout << endl;

  fstream fs;
  string line;

  string filename = "/"+ cmd;
  cout << "The string filename is: " << filename << endl;

  string pathName = FILE_PATH + filename;
  cout << "The string pathName is: " << pathName << endl;
  cout << endl;

  // The c_str() method returns a C++ string as a C string.
  fs.open((pathName).c_str(), fstream::in);
  
  //fs.open((FILE_PATH + filename).c_str(), std::fstream::in);
					
  while(getline(fs,line)) cout << line << endl;
  fs.close();
					    
}
