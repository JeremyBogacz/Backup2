#!/bin/bash

# Created by Jeremy Bogacz on 10/6/2021

# CPE 422 homework 4 question 2

# Test to see if 1>2, if yes subtract 1-2 and print message.
# If no, subtract 2-1 and print message.
if [ $1 -gt $2 ]
then
	diff=$(($1-$2))
	echo "$1 is greater than $2."
else
	diff=$(($2-$1))
	echo "$2 is greater than $1."
fi

# Print the difference caluclated above to the screen.
echo "Difference = $diff"
echo

# Count down from the difference to 1 printing to the screen each time.
counter=$diff

while [ $counter -gt 0 ]
do
	echo "Count down = $counter"
	((counter--))
done



