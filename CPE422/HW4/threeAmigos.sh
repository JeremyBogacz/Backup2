#!/bin/bash

# Created by Jeremy Bogacz on 10/6/2021

# CPE 422 homework 4 question 3

# Calculate sum of first three inputs.
sum=$(($1+$2+$3))
echo "Sum = $sum"

# Call subtractMachine and pass the sum and the fourth input.
./subtractMachine.sh $sum $4
