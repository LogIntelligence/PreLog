# PCLLog
Code for "Pre-training with Contrastive Learning for Unified Log Analytics"


## 2. Requirements
### 2.1. Environment
- Python 3.8
- torch
- transformers
- ...

Installation guide:
```shell
$ pip install -r requirements.txt
$ cd fairseq && python setup.py install
```

### 2.2. Models and data
Download and upzip checkpoint for pre-training, a small set of pre-training data, and the pre-trained PCLLog [here](https://figshare.com/s/b62ffa904644863a2b89).

## 3. Usage
### 3.1. Pre-training PCLLog
Pre-training with a small set of data
```shell
$ ./scripts/preprocess.sh
$ ./script/pretrain.sh
```

### 3.2. Prompt Tuning PCLLog
#### 3.2.1. Generation Task
Example of Log Parsing as Generation on HDFS:
```shell
$ cd tasks/generation/logparsing
$ python train.py --dataset HDFS \
    --model-path ../../../models/PCLLog \
    --train-file ./datasets/HDFS/32shot/1.json \
    --test-file ./datasets/HDFS/test.json \
    --outdir 32shot/itr_1/PCLLog
```

Run benchmark on 16 datasets:
```shell
$ cd tasks/generation/logparsing
$ ./benchmark
```
#### 3.2.2. Classification Task
Example of Anomaly Detection as Classification on BGL with a small set of data:
```shell
$ cd task/classification
$ python train.py \
    --dataset BGL \
    --model-path ../../models/PCLLog \
    --train-file anomaly_detection/data/train.json \
    --test-file anomaly_detection/data/test.json \
    --prompt-template prompt_template.txt \
    --verbalizer anomaly_detection/verbalizer.txt 
```


Example of Failure Identification as Classification on BGL with a small set of data:
```shell
$ cd task/classification
$ python train.py \
    --dataset OpenStack \
    --model-path ../../models/PCLLog \
    --train-file failure_identification/data/train.json \
    --test-file failure_identification/data/test.json \
    --prompt-template prompt_template.txt \
    --verbalizer failure_identification/verbalizer.txt
```

Full datasets for anomaly detection and failure identification can be found here.

## 4. Results
### 4.1. RQ1


### 4.2. RQ2


### 4.3. RQ3


### 4.4. RQ4