#!/bin/bash
# cd ../..; # the python scripts are all in the root directory.
echo ${which python};
echo $PWD;
echo "previewing args";
echo $1 -S $2 -M $3 -TS $4 -BS $5 -GA $6 -LM $7 -SP $8 $9 ${10} ${11} ${12} ${13} # last 5 are boolean flags
python main.py $1 -S $2 -M $3 -TS $4 -BS $5 -GA $6 -LM $7 -SP $8 $9 ${10} ${11} ${12} ${13}; # last 5 are boolean flags