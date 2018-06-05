# Chinese poem writer

### Introduction
We built a Chinese poem writer based on [Temporal Convolutional Networks (TCN)](https://arxiv.org/abs/1803.01271). This paper recently indicates that a simple TCN architecture outperforms RNNs across a diverse range of tasks and datasets, while demonstrating longer effective memory.
Therefore, we built this Chinese poem writer using TCN.

### Dependencies
```
Python3.x (Tested with 3.5)
PyTorch (Tested with 0.4.0)
```

### Installation
To use this code, please do:


0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/Chinese_Poem_Writer
    cd Chinese_Poem_Writer
    ```
1. To try the demo code, please run:
    ```Shell
    python main.py -m fast
    ```
### Training
To train this Chinese poems writer, you can run:

    python train.py

For more details, please see in [train.py](https://github.com/jfzhang95/Chinese_Poem_Writer/blob/master/train.py).

### Usage
We built three poem generator modes:

0. fast - it can generate each word only based on the last word.
1. context - it can generate each word based on the entire previous context.
2. head - it can generate Acrostic based on user inputs.

To change generator mode, you should change "fast" to "context" or "head" behind:

    python main.py -m fast

### Results

    python main.py -m fast

<img src="https://github.com/jfzhang95/Chinese_Poem_Writer/blob/master/doc/fast.png" width = "450" height = "120" alt="fast" />

    python main.py -m context

<img src="https://github.com/jfzhang95/Chinese_Poem_Writer/blob/master/doc/context.png" width = "450" height = "120" alt="context" />

    python main.py -m head

<img src="https://github.com/jfzhang95/Chinese_Poem_Writer/blob/master/doc/head.png" width = "450" height = "120" alt="head" />

Upload generated poem into [diyiziti](http://www.diyiziti.com/), we can get more artistic result!

![4](doc/demo.png)

### TODO

- [x] Basic model and function
- [ ] Write a script to upload generated poem automatically
- [ ] Training our model with more and better [data](https://github.com/chinese-poetry/chinese-poetry)


We thank the authors of [pytorch-tcn](https://github.com/locuslab/TCN) for making their PyTorch implementation of TCN available!
