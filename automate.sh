#!/bin/bash

modelName=$1
modelWhich=$2
personA=$3
personB=$4

# Create dirs for models and output
mkdir -p models/$modelName/
mkdir -p output/$modelName/
mkdir -p output/${modelName}-swap/
# inputA is photo/$personA/
# inputB is photo/$personB/

# Create log files
line="Model: ./models/${modelName}/ %nInputs: ./photo/${personA}/, ./photo/${personB}/ %nOutputs: ./output/${modelName}/, ./output/${modelName}-swap/ %nTimestamp: $(date "+$D") $(date "+%T")"
$line >> models/$modelName/log.md

# Call extracting command
#python3 faceswap.py extract -i photo/$personA/ -o data/$personA/
#python3 faceswap.py extract -i photo/$personB/ -o data/$personB/

# Call training command
python3 faceswap.py train -A data/$personA/ -B data/$personB/ -m models/$modelName/ -w -t $modelWhich

line="Log: ./models/log.md %nTimestamp: $(date "+%D") $(date "+%T")"
$line >> models/$modelName/log.md

# Call converting command
python3 faceswap.py convert -i photo/$personA -o output/$modelName/ -m models/$modelName
python3 faceswap.py convert -i photo/$personB -o output/${modelName}-swap/ -m models/$modelName -s

# View output
#xdg-open output
