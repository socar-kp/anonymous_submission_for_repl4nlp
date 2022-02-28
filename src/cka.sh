#!bin/bash
set -e

pair="bert,bert bart,bart roberta,roberta electra,electra"

for i in $pair
do
    IFS=','
    set -- $i
    echo $1 and $2
    python cka_module.py --model1 $1 --model2 $2
    
done