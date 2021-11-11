#include<iostream>
#include <stdlib.h>
int main(int argc, char *argv[]){
// This program takes one command line argument which is a string.
// The program prints out the string to the Linux shell along with
// the name of the program.
// Print out a usage message if incorrect number of arguments.
if(argc == 1){
	 std::cout << "Usage: ./hello_3 number." << std::endl;
	 std::cout << "Where number is an integer." << std::endl;
}
if(argc == 2) {
	 std::cout << "argv[0] is: " << argv[0] << std::endl;
	 std::cout << "argv[1] is: " << argv[1] << std::endl;
	 std::cout << std::endl;
	 for(int i=0; i< atoi(argv[1]); ++i) {
	 	std::cout << "Hello World!" << std::endl;
	 }
}
 return 0;
}
