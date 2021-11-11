/** A program that opens a text file, write some txt to the file and then closes it.
 * *   The text file name is passed on the command line as an argument.
 * */


#include<iostream>
#include<fstream>
#include<string>
using namespace std;

#define FILE_PATH "/home/debian/CPE422"


int main(int argc, char* argv[]){
  if(argc!=2){
	  cout << endl;
	  cout << endl;
	  cout << "Usage is writing_file and one name of file to be written." << endl;
	  cout << " e.g. writing-file movies.txt" << endl;
	  return 2;
  }
  string cmd(argv[1]);
  
  cout << endl; 
  cout << endl;
  cout << "Starting the writing-file program" << endl;
  cout << "The current FILE Path is: " << FILE_PATH << endl;
  cout << "The file name passed is: " << cmd << endl;
		        
  fstream fs;
  string path(FILE_PATH);
  fs.open((path+"/"+cmd).c_str(), std::fstream::out);
  fs << "I watched the movie Star Wars I." << endl;
  fs << "I watched the movie Star Trek." << endl;
  fs.close();

  cout << "Finished the writing-file Program" << endl;
  cout << endl;
  cout << endl;
  return 0;
}

