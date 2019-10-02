tree2code: Learning Discrete Syntactic Codes for Structural Diverse Translation
===

This code implements the syntactic code learning model in paper "Generating Diverse Translation with Sentence Codes" published in ACL 2019.
PDF of the paper can be acquired here: https://www.aclweb.org/anthology/P19-1177/ . Citation:

```
@inproceedings{Shu2019GeneratingDT,
  title={Generating Diverse Translations with Sentence Codes},
  author={Raphael Shu and Hideki Nakayama and Kyunghyun Cho},
  booktitle={ACL},
  year={2019}
}
```

The motivation of learning syntactic codes is to extract the syntactic structure of a sentence, and encode the structural choice into discrete codes.
Then if we merge the syntactic codes with the target tokens and train a neural machine translation model to predict both of them, the code can effectively condition of the generation of the whole translation.

More concretely, suppose "\<c15\>" represents the syntactic structure of the sentence "John hates school.". 
Then we are going to train a model that does this kind of translation:
```
John odia la escuela. -> <c15> John hates school.
```

So, how can we encode syntax into discrete codes? Well, we combine the ideas of TreeLSTM autoencoder and discretization bottleneck.
The TreeLSTM autoencoder creates a representation of a parse tree, and the bottleneck quantize the vector representation into discrete numbers, which is quite straight-forward.

<img src="https://i.imgur.com/DjrFF70.png" width="600px"/>

Now, this is the model, it has a pretty clear architecture of TreeLSTM-based autoencoder.
The only thing you may be surprised is that we also feed the context of the source sequence into the autoencoder. Why we do this?
The reason is simple. Without the source context, you are literally trying to encode almost infinite variations of parse trees into say only 32 codes, which is an impossible task even for Captain America.
With the source information, you are just encoding the *choice* of syntactic structures.

The TreeLSTM part may looks complicated, but fortunately it can be efficiently implemented by DGL (Deep Graph Library), which is developed by NYU and AWS team.

## Install Package Dependency

The code depends on PyTorch, **dgl** for TreeLSTM,
**nltk**, **torchtext** and **networkx** for tree loading,
 **nmtlab** for Transformer encoder and **horovod** for multi-gpu training.

We recommend installing with conda.

-1. (If you don't have conda) Download and Install Miniconda for Python 3

```
mkdir ~/apps; cd ~/apps
wget  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Reload the bash/zsh and run `python` to check it's using the python in Miniconda.

-2. Install pytorch following https://pytorch.org/get-started/locally/

-3. Install dgl following https://www.dgl.ai/pages/start.html

-4. (Only for multi-gpu training) Install horovod following https://github.com/horovod/horovod#install

```
mkdir ~/apps; cd ~/apps
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar xzvf openmpi-4.0.1.tar.gz
cd openmpi-4.0.1
# Suppose you have Miniconda3 in your home directory
./configure --prefix=$HOME/miniconda3 --disable-mca-dso
make -j 8
make install
```

Check whether the openmpi is correctly installed by running `mpirun`. Then install horovod with:

```
conda install -y gxx_linux-64
# If you don't have NCCL
pip install horovod
# If you have NCCL in /usr/local/nccl
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/usr/local/nccl pip install horovod
```

Check horovod by running `horovodrun`.

-5. Run `pip install nltk networkx torchtext nmtlab`

-6. Clone this github repo, run 
```
cd ~/
git clone https://github.com/zomux/tree2code
cd tree2code
```

## Download pre-processed WMT14 dataset with parsed CFG trees

-1. Create `mydata` folder if it's not there
```
mkdir mydata
cd mydata
```

-2. Download pre-processed WMT14 dataset from https://drive.google.com/file/d/1QI_tEs7xQLgwCbvkbpHynIaBvfFCccPz/view .
After download, uncompress the dataset in side `mydata` folder.
```
./gdown.pl https://drive.google.com/file/d/1QI_tEs7xQLgwCbvkbpHynIaBvfFCccPz/view tree2code_wmt14.tgz
tar xzvf tree2code_wmt14.tgz
```

## Train the model

-1. Go back to `tree2code` folder

-2. (Single GPU) Run this command:
```
python run.py --opt_dtok wmt14 --opt_codebits 8 --opt_limit_tree_depth 2 --opt_limit_datapoints 100000 --train
```

-2. (Multi-GPU) Run this command if you have 8 GPUs:
```
horovodrun -np 8 -H localhost:8 python run.py --opt_dtok wmt14 --opt_codebits 8 --opt_limit_tree_depth 2 --opt_limit_datapoints 100000 --train
```

There are some options you can use for training the model.

``--opt_codebits`` specifies the number of bits for each discrete code, 8 bits means 256 categories

``--opt_limit_tree_depth`` limit the depth of a parse tree to consider. The model will consider up to three tree layers when we limit the depth to 2.
You can increase the depth and monitor the `label_accuracy` to ensure the reconstruction accuracy is not too low.

``--opt_limit_datapoints`` limit the number of training datapoints to be used on each GPU as training on the whole dataset is time-consuming.
In our experiments, we train the model with 8 GPUs, and limit the training datapoints to 100k on each GPU, which results in an effective training dataset of 800k samples.

The script will train the model for 20 epochs, and you will see the increasing performance like this:
```
[nmtlab] Training TreeAutoEncoder with 74 parameters
[nmtlab] with Adagrad and SimpleScheduler
[nmtlab] Training data has 773 batches
[nmtlab] Running with 8 GPUs (Tesla V100-SXM2-32GB)
[valid] loss=6.62 label_accuracy=0.00 * (epoch 1, step 1)
[valid] loss=1.43 label_accuracy=0.54 * (epoch 1, step 194)
...
[nmtlab] Ending epoch 1, spent 2 minutes
...
[valid] loss=0.47 label_accuracy=0.86 * (epoch 12, step 14687)
...
```

## Export the syntactic codes for all training samples

Once we obtain the syntactic coding model, we need to get the syntactic codes for all training samples. Just run this command:

```
python run.py --opt_dtok wmt14 --opt_codebits 8 --opt_limit_tree_depth 2 --opt_limit_datapoints 100000 --export_code
```

It's going to take time as the dataset is large. Once it's done, if you go to `mydata/tree2code_codebits-8_dtok-wmt14_limit_datapoints-100000_limit_tree_depth-2.codes`,
you will see things like this:

> ▁Der ▁Bau ▁und ▁die ▁Reparatur ▁der ▁Auto straßen ...   (ROOT (NP (NP (NN Construction) (CC and) (NN repair)) (PP (IN of) (NP (NNP highways) (NNP and))) (: ...)))      246

Each line has three parts: source sequence, target-side tree and syntactic code.

## Merge the codes with target sentences in the training set

Finally, we merge the generated codes with target sentences in the training set.
```
python run.py --opt_dtok wmt14 --opt_codebits 8 --opt_limit_tree_depth 2 --opt_limit_datapoints 100000 --make_target
```

If you go to `mydata/tree2code_codebits-8_dtok-wmt14_limit_datapoints-100000_limit_tree_depth-2.tgt`, you can find the file like this:

> \<c247\> \<eoc\> ▁Construction ▁and ▁repair ▁of ▁highway s ▁and ... 

## Result files

You can also find all the result files here:

https://drive.google.com/drive/folders/1w6bo30D3VaoIoVAv6fTFbxfMNtDJwczr

The file `tree2code_codebits-8_dtok-wmt14_limit_datapoints-100000_limit_tree_depth-2.tgt` is the target-side training data with prefixing codes.

## Todos

- [x] put nmtlab on pypi
- [ ] Discuss the code inbalance issue
- [ ] add the script for training NMT models and sample diverse translations