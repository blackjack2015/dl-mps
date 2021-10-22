#!/bin/bash

nets=("resnet50")
bses=(4)
replicas=(1 2 4)

for net in "${nets[@]}"
do
    for bs in "${bses[@]}"
    do
        for repli in "${replicas[@]}"
        do

            echo $net $bs temporal $repli
            echo quit | nvidia-cuda-mps-control
            sleep 5
            python batch.py --replicas $repli --net $net --batch-size $bs --repetitions 300 --mode temporal
            
        done
    done
done

