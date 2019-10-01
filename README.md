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

The code depends on PyTorch, **dgl** for TreeLSTM, **nmtlab** for Transformer encoder and **horovod** for multi-gpu training.

We recommend installing with conda.

1. Install pytorch following https://pytorch.org/get-started/locally/

2. Install dgl following https://www.dgl.ai/pages/start.html

3. (Only for multi-gpu training) Install horovod following https://github.com/horovod/horovod#install

4. Run `pip install nmtlab`

5. Clone this github repo, run 
```
git clone https://github.com/zomux/tree2code
cd tree2code
```

## Download pre-processed WMT14 dataset with parsed CFG trees

-1. Create `mydata` folder, run
```
mkdir mydata
cd mydata
```

-2. Download pre-processed WMT14 dataset from https://drive.google.com/file/d/1QI_tEs7xQLgwCbvkbpHynIaBvfFCccPz/view

-3. Uncompress the dataset in side `mydata` folder
```
tar xzvf tree2code_wmt14.tgz
```

## Train the model

-1. Go back to `tree2code` folder

-2. (Single GPU) Run this command:
```
python run.py --opt_dtok wmt14 --opt_codebits 5 --opt_limit_tree_depth 2 --opt_limit_datapoints 10000 --train
```

-2. (Multi-GPU) Run this command if you have 4 GPUs:
```
horovodrun -np 4 -H localhost:4 python run.py --opt_dtok wmt14 --opt_codebits 5 --opt_limit_tree_depth 2 --opt_limit_datapoints 10000 --train
```

There are some options you can use for training the model.

``--opt_codebits`` specifies the number of bits for each discrete code, 5 bits means 32 categories

``--opt_limit_tree_depth`` limit the depth of a parse tree to consider. The model will consider up to three tree layers when we limit the depth to 2.
You can increase the depth and monitor the `label_accuracy` to ensure the reconstruction accuracy is not too low.

``--opt_limit_datapoints`` limit the number of training datapoints to be used on each GPU as training on the whole dataset is time-consuming.
In our experiments, we train the model with 8 GPUs, we limit the training datapoints to 100k on each GPU, which results in an effective training dataset of 800k samples.

## Export the syntactic codes for all training samples

Run
```
python run.py --opt_dtok wmt14 --opt_codebits 5 --opt_limit_tree_depth 2 --opt_limit_datapoints 10000 --export_code
```

## Merge the codes with target sentences in the training set
Run
```
python run.py --opt_dtok wmt14 --opt_codebits 5 --opt_limit_tree_depth 2 --opt_limit_datapoints 10000 --make_target
```

## Todos

- [ ] put nmtlab on pypi
- [ ] add the script for training NMT models and sample diverse translations