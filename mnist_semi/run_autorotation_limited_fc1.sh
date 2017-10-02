#!/bin/bash

degree=10
while [ $degree -le 90 ]
do
    counter=1
    file_name="test_autorotation_fc1_$degree.txt"
    while [ $counter -le 5 ]
    do
        python autorotation_limited_fc1.py $degree >> $file_name
        ((counter++))
    done
    echo "$counter done"
    degree=$(($degree + 10))
done 
