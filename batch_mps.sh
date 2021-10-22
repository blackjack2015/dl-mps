#!/bin/bash

nets=("resnet18" "resnet50")
bses=(4 16)
replicas=(1 2 4)
repes=400

nvidia-cuda-mps-control -d

for net in "${nets[@]}"
do
    for bs in "${bses[@]}"
    do
        for repli in "${replicas[@]}"
        do

            echo $net $bs mps $repli
            sleep 5
            python batch.py --replicas $repli --net $net --batch-size $bs --repetitions $repes --mode mps
            
        done
    done
done

