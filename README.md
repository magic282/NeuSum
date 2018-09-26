# NeuSum
This repository contains code for the ACL 2018 paper "Neural Document Summarization by Jointly Learning to Score and Select Sentences"

## About this code
**PyTorch version**: This code requires PyTorch v0.3.x.

**Python version**: This code requires Python3.


## How to run

### Prepare the dataset and code
Make a folder for the code and data:
```bash
NEUSUM_HOME=~/workspace/neusum
mkdir -p $NEUSUM_HOME/code
cd $NEUSUM_HOME/code
git clone --recursive https://github.com/magic282/NeuSum.git
```
After preparation, the workspace looks like:
```
neusum
├── code
│   └── NeuSum
│       └── neusum_pt
│           ├── neusum
│           └── PyRouge
└── data
    └── cnndm
        ├── dev
        ├── glove
        ├── models
        └── train
```

The paper used CNN / Daily Mail dataset.
 
[About the CNN Daily Mail Dataset](https://gist.github.com/magic282/e4b2ecc91f185939b2688863ae9e41c1)

[About the CNN Daily Mail Dataset 2](https://github.com/magic282/cnndm_acl18)
### Setup the environment
#### Package Requirements:
```
nltk numpy pytorch
```
**Warning**: Older versions of NLTK have a bug in the PorterStemmer. Therefore, a fresh installation or update of NLTK is recommended.

A Docker image is also provided.
#### Docker image
```bash
docker pull magic282/pytorch:0.3.0
```
### Run training
The file `run.sh` is an example. Modify it according to your configuration.
#### Without Docker
```bash
bash $NEUSUM_HOME/code/NeuSum/neusum_pt/run.sh $NEUSUM_HOME/data/cnndm $NEUSUM_HOME/code/NeuSum/neusum_pt
```
#### With Docker
```bash
nvidia-docker run --rm -ti -v $NEUSUM_HOME:/workspace magic282/pytorch:0.3.0
```
Then inside the docker:
```bash
bash code/NeuSum/neusum_pt/run.sh /workspace/data/cnndm /workspace/code/NeuSum/neusum_pt
```