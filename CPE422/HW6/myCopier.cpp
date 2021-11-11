// Written by Jeremy Bogacz on 10/23/2021
//
// This program takes in two file names and will copy
// the contents of the first to file to the second file.
//
// Example invocation: ./myCopiercpp myfile1.txt myfile2.txt

#include<iostream>
#include<fstream>
#include<string>
using namespace std;

#define FILE_NAME "/home/debian/CPE422/HW6"

int main(int argc, char* argv[]){
	if(argc!=3){
		cout << endl;
		cout << "Usage is myCopier, name of file to be read, name of file to be written:" << endl;
		cout << "e.g. myCopier file1.txt file2.txt" << endl;
		return 2;
	}

	cout << endl;
	cout << "Starting myCopier.cpp" << endl;
	
	// Save name of the two files
	string cmd_1(argv[1]);
	string cmd_2(argv[2]);

	fstream fs_1;
	fstream fs_2;

	string path(FILE_NAME);

	// Build the absolute locations of each file
	string pathName_1 = path + "/" + cmd_1;
	string pathName_2 = path + "/" + cmd_2;

	// Open file 1 for reading and file 2 for writting
	fs_1.open((pathName_1).c_str(), fstream::in);
	fs_2.open((pathName_2).c_str(), fstream::out);

	cout << endl;
	cout << "Read File Location: " << pathName_1 << endl;
	cout << "Write File Location: " << pathName_2 << endl;
	cout << endl;

	string line;

	cout << "Copying..." << endl;

	// Copy the contents of file 1 to file 2
	while(getline(fs_1,line)) fs_2 << line << endl;

	cout << "Done." << endl << endl;

	// Close both files
	fs_1.close();
	fs_2.close();

	cout << "myCopier.cpp finished." << endl << endl;

	return 0;

}
