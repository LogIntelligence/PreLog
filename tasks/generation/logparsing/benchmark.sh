#!/bin/bash

MODEL_NAME="../../../models/PCLLog"
shot=32
for rtime in 1 2 3 4 5; do
  for dataset in Android Apache BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper; do
    echo "${rtime} - ${shot} - ${dataset}"
    python train.py \
    --dataset ${dataset} \
    --model-path $MODEL_NAME \
    --train-file ./datasets/${dataset}/${shot}shot/${rtime}.json \
    --test-file ./datasets/${dataset}/test.json \
    --outdir ${shot}/${rtime}/PCLLog
  done
done
