#!/bin/bash
configFile=$1
numProcc=$2

if [ -z $configFile ]; then
	echo "There is not configuration file"
	echo "Goodbye."
elif [ -z $numProcc ]; then
	echo "There is not number of processes to generate data. Basic case is 1"
	echo "Goodbye."
else
	echo "Creating tmux model"
	tmux new -d -s model
	tmux send-keys -t model.0 "python Constructors.py $configFile" ENTER
	tmux a -t model

	for (( n=0; n<$numProcc; n++ ))
	do
		data_gen="data_generator_"$n
		echo "Creating tmux "$data_gen
		tmux new -d -s $data_gen
		tmux send-keys -t $data_gen.0 "python data_generator.py $configFile $n $numProcc" ENTER
		sleep 2
	done
fi
