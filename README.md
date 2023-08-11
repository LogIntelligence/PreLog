# PreLog: Pre-training for Log Analytics

**Abstract**: Large-scale software-intensive systems often produce a large volume of logs to record runtime status and
events for troubleshooting purposes. The rich information included in log data enables a variety of log analytics tasks,
such as log parsing and anomaly detection. Over the years, many approaches have been proposed for automated log
analytics. However, these approaches usually design separate models for each specific task, which cannot be generalized
to other tasks. They are also not robust when dealing with logs from heterogeneous sources. In this paper, we propose
PreLog, a novel pre-trained sequence-to-sequence model for log analytics. PreLog is pre-trained on a large amount of
unlabelled log data to capture the semantic meaning of logs. We design two log-specific pre-training objectives,
including entry-level and sequence-level objectives, which enable PreLog to better understand the hidden structure and
semantic meaning of logs. To perform downstream tasks, we leverage a prompt tuning paradigm to convert downstream tasks’
objectives into a similar form as the pre-training stage. We have conducted extensive experiments on two main downstream
tasks (i.e., log parsing and log-based anomaly detection). Experimental results show that PreLog achieves better or
comparable results in comparison with the state-of-the-art, task-specific approaches. PreLog is cost-effective and can
be uniformly applied to many log analytics tasks through a prompt tuning paradigm.

## 1. Framework

<p align="center"><img src="docs/images/architecture.png" width="1000"><br>An overview of PreLog</p>

## 2. Requirements

### 2.1. Environment

- Python 3.8
- torch
- transformers
- accelerate
- ...

Installation guide:

```shell
$ pip install -r requirements.txt
$ cd fairseq && python setup.py install
```

### 2.2. Models and data

Download and unzip checkpoint for pre-training, a small set of pre-training data, and the pre-trained
PreLog [here](https://figshare.com/s/5a08ef8b02b94f6726c2).

## 3. Usage

### 3.1. Pre-training PreLog

- Tokenize and binarize data:

```shell
# set path to raw pre-training data (DATA_DIR) in scripts/preprocess.sh and run
$ ./scripts/preprocess.sh
```

- Pre-train PreLog:

```shell
# set path to tokenized pre-training data (DATA_DIR), path to save model (SAVE_DIR), checkpoint (CHECKPOINT_PATH) in scripts/pretrain.sh and run 
$ ./scripts/pretrain.sh
```

- Convert checkpoint to huggingface format:

```shell
# set path to save model (CHECKPOINT_PATH) in scripts/convert_fairseq_to_hf.py and run
$ python ./scripts/convert_fairseq_to_hf.py
```

### 3.2. Prompt Tuning PreLog

#### 3.2.1. Generation Task

- Log Parsing as Generation:

  **Dataset:** We use the [corrected](https://dl.acm.org/doi/abs/10.1145/3510003.3510101) version originated
  from [LogPAI benchmark](https://github.com/logpai/logparser) with 16 datasets. The statistics of these datasets are as
  follows:

|   **Dataset**   | Spark | OpenStack | Windows | Apache | OpenSSH | Proxifier | HealthApp | Thunderbird | HPC | Android | HDFS | BGL | Zookeeper | Mac | Hadoop | Linux |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| **\#Templates** | 36 | 43 | 50 | 6 | 26 | 8 | 75 | 149 | 45 | 158 | 14 | 120 | 50 | 341 | 114 | 116 |

```shell
# to run on HDFS dataset
$ cd tasks/generation/logparsing
$ export MODEL_PATH="path to PreLog model"
$ accelerate launch train.py \
    --dataset HDFS \
    --model-path $MODEL_PATH \
    --train-file data/HDFS/32shot/1.json \
    --test-file data/HDFS/test.json \
    --outdir parsing_hdfs
```

- Run benchmark on 16 datasets:

```shell
# set path to PreLog model (MODEL_PATH) in tasks/generation/benchmark.sh and run
$ cd tasks/generation
$ ./benchmark.sh
```

#### 3.2.2. Classification Task

- Anomaly Detection as Classification:

  **Datasets:** We use commonly-used HDFS, BGL, Spirit datasets (from [[1]](https://doi.org/10.1109/ASE51524.2021.9678773), [[2]](http://doi.org/10.1145/3510003.3510155)). The statistics of these datasets are shown in the following table:

|                  | **Category**       | **Size** | **\#Messages** | **\#Anomalies** |
| :--------------: | :----------------: | :------: | :------------: | :-------------: |
| **HDFS**         | Distributed system | 1\.5 G   | 11,175,629     | 16,838          |
| **Blue Gene /L** | Supercomputer      | 743 M    | 4,747,963      | 348,460         |
| **Spirit**       | Supercomputer      | 1\.0 G   | 7,983,345      | 768,142         |

```shell
$ cd task/classification
$ export MODEL_PATH="path to PreLog model"
$ export DATADIR="path to anomaly detection data"
$ accelerate launch train.py \
    --dataset BGL \
    --model-path $MODEL_PATH \
    --train-file $DATADIR/train.json \
    --test-file $DATADIR/test.json \
    --prompt-template prompt_template.txt \
    --verbalizer anomaly_detection/verbalizer.txt \
    --batch-size 16 \
    --lr 3e-5 \
    --max-step 2000 \
    --lr-scheduler-type polynomial \
    --max-length 1024 \
    --do-train \
    --do-eval
```

- Failure Identification as Classification:

  **Dataset**: We adopt the OpenStack dataset from [[3]](https://doi.org/10.1002/spe.3164). This dataset contains 3 types
  of failures, including:
1. VM is destroyed ungracefully right after creation and before completely go through its life cycle;
2. After the creation of the VM, its virtual disk is removed from the host server. Unlike the former anomaly where the VM is destroyed, the VM configuration remains unchanged, though it does not have access to the storage space required for booting the operating system;
3. A disturbance is applied to the performance of Neutron, which is responsible for managing the network. In this way, decreasing the responsiveness time of this component led to the timeout error, and also by stopping the DHCP service that is charged for assigning IP to VM, the VM network is disturbed practically

|                          | **\#Sequences** |
|:------------------------:|:---------------:|
|   **VM is destroyed**    |       167       |
| **VM's disk is removed** |       225       |
| **Network disturbance**  |       169       |

```shell
$ cd task/classification
$ export MODEL_PATH="path to PreLog model"
$ export DATADIR="path to failure identification data"
$ accelerate launch train.py \
    --dataset BGL \
    --model-path $MODEL_PATH \
    --train-file $DATADIR/train.json \
    --test-file $DATADIR/test.json \
    --prompt-template prompt_template.txt \
    --verbalizer failure_identification/verbalizer.txt \
    --batch-size 16 \
    --lr 3e-5 \
    --max-step 2000 \
    --lr-scheduler-type polynomial \
    --max-length 1024 \
    --do-train \
    --do-eval
```

## 4. Results

### 4.1. RQ1: Log Parsing

We evaluate the accuracy of log parsing performed by PreLog. We compare PreLog with the top-performing data-driven log parsers, i.e., Spell, Drain, Logram, and SPINE; and the current state-of-the-art DL-based log parser, i.e., LogPPT.

- Source code for Spell, Drain, and Logram is adopted from [LogPAI](https://github.com/logpai/logparser) and [empirical study](https://figshare.com/articles/software/Artifact_for_Guidelines_for_Assessing_the_Accuracy_of_Log_Message_Template_Identification_Techniques_/18858332).
- We use the implementation provided by authors for [SPINE](https://doi.org/10.1145/3540250.3549176).
- Source for LogPPT is adopted from [LogPPT](https://github.com/LogIntelligence/LogPPT).

<p align="center"><img src="docs/images/RQ1-accuracy.png" width="1000"></p>

<p align="center"><img src="docs/images/RQ1-robustness.png" width="500"></p>

**Take-away points:**
- PreLog achieves the best GA on 9 out of 16 datasets, a GA of over 0.9 on 12 datasets and 1.0 accuracy on seven datasets.
- PreLog significantly outperforms data-driven log parsers and achieves comparable results with LogPPT.
- The performance of PreLog can be improved if more labelled samples are provided, and it can achieve good results with 16 or more labelled samples.

### 4.2. RQ2: Anomaly Detection

We compare PreLog with CNN, LogRobust, and NeuralLog, which are the state-of-the-arts on anomaly detection.

- **_With Stable Logs_**:

<p align="center"><img src="docs/images/RQ2-stable.png" width="500"></p>

- **_With Unstable Log Events_**:

<p align="center"><img src="docs/images/RQ2-events.png" width="500"></p>

- **_With UnStable Log Sequences_**:

<p align="center"><img src="docs/images/RQ2-sequences.png" width="500"></p>

**Take-away points:**
- PreLog can capture the semantic meaning of log sequences more effectively via pre-training on a large amount of data, thus leading to the better results compared to the state-of-the-art.
- PreLog maintains a consistently high accuracy (F-measure ranging from 0.936 to 0.942 with unstable log events and from 0.936 to 0.950 with unstable log sequences) under high injection ratios.
- PreLog is effective and robust for log-based anomaly detection on both stable and unstable log data.

### 4.3. RQ3: Ablation Study
We evaluate the effectiveness of each pre-training objective when the model is trained without it.
- **_Log Parsing_**:

<p align="center"><img src="docs/images/RQ3-parsing.png" width="500"></p>

- **_Anomaly Detection with Stable Logs_**:

<p align="center"><img src="docs/images/RQ3-AD-stable.png" width="500"></p>

- **_Anomaly Detection with Unstable Logs_**:
    - _Unstable Log Events_:
      <p align="center"><img src="docs/images/RQ3-AD-event.png" width="500"></p>
    - _Unstable Log Sequences_
      <p align="center"><img src="docs/images/RQ3-ad-seq.png" width="500"></p>

**Take-away points:**
- Pre-training with both entry-level and sequence-level objectives is important for log parsing.
- PreLog performs worse when one of the pre-training objectives is excluded on unstable log data.
    
### 4.4. Other log analytics tasks

- Failure Identification on OpenStack:

    We ask PreLog to identify the failure types of OpenStack system. The results show that PreLog can achieve an F-measure of over 0.95 for all failure types on the OpenStack dataset.

<p align="center"><img src="docs/images/failure-identification.png" width="500"></p>

- Fault-indicated Event Identification:
    
    We ask PreLog to locate the logs in log sequence that are most likely to cause anomalies. By leveraging attention scores assigned for each log message, PreLog can locate anomalies in log sequences with high accuracy, especially on the Spirit dataset.

<p align="center"><img src="docs/images/inference.png" width="500"></p>

