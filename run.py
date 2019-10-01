#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os, sys
from torch import optim
from argparse import ArgumentParser
sys.path.append(".")

import torch
from nmtlab import MTTrainer
from nmtlab.utils import OPTS
from nmtlab.utils import is_root_node

from lib_treeautoencoder import TreeAutoEncoder
from lib_treelstm_dataloader import BilingualTreeDataLoader
from datasets import get_dataset_paths

DATA_ROOT = "./mydata"

ap = ArgumentParser()
ap.add_argument("--resume", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--test_nbest", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--export_code", action="store_true")
ap.add_argument("--make_target", action="store_true")
ap.add_argument("--make_oracle_codes", action="store_true")
ap.add_argument("--all", action="store_true")
ap.add_argument("--opt_dtok", default="aspec", type=str)
ap.add_argument("--opt_seed", type=int, default=3)
ap.add_argument("--opt_hiddensz", type=int, default=256)
ap.add_argument("--opt_without_source", action="store_true")
ap.add_argument("--opt_codebits", type=int, default=0)
ap.add_argument("--opt_limit_tree_depth", type=int, default=0)
ap.add_argument("--opt_limit_datapoints", type=int, default=-1)
ap.add_argument("--model_path",
                default="{}/tree2code.pt".format(DATA_ROOT))
ap.add_argument("--result_path",
                default="{}/tree2code.result".format(DATA_ROOT))
OPTS.parse(ap)

n_valid_per_epoch = 4

# Define datasets
DATA_ROOT = "./mydata"
dataset_paths = get_dataset_paths(DATA_ROOT, OPTS.dtok)

# Using horovod for training, automatically occupy all GPUs
# Determine the local rank
if torch.cuda.is_available():
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    part_index = hvd.rank()
    part_num = hvd.size()
    gpu_num = hvd.size()
else:
    part_index = 0
    part_num = 1
    gpu_num = 1
print("Running on {} GPUs".format(gpu_num))

# Define dataset
dataset = BilingualTreeDataLoader(
    src_path=dataset_paths["train_src_corpus"],
    cfg_path=dataset_paths["train_cfg_corpus"],
    src_vocab_path=dataset_paths["src_vocab_path"],
    treelstm_vocab_path=dataset_paths["cfg_vocab_path"],
    cache_path=None,
    batch_size=128 * gpu_num,
    part_index=part_index,
    part_num=part_num,
    max_tokens=60,
    limit_datapoints=OPTS.limit_datapoints
)

# Load the tree autoencoder onto GPU
autoencoder = TreeAutoEncoder(dataset, hidden_size=OPTS.hiddensz, code_bits=OPTS.codebits, without_source=OPTS.without_source)
if torch.cuda.is_available():
    autoencoder.cuda()

# Train the model
if OPTS.train or OPTS.all:
    # Training code
    from nmtlab.schedulers import SimpleScheduler
    scheduler = SimpleScheduler(30)
    weight_decay = 1e-5 if OPTS.weightdecay else 0
    optimizer = optim.Adagrad(autoencoder.parameters(), lr=0.05)
    trainer = MTTrainer(autoencoder, dataset, optimizer, scheduler=scheduler, multigpu=gpu_num > 1)
    OPTS.trainer = trainer
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria="loss",
    )
    if OPTS.w_pretrain:
        import re
        pretrain_path = re.sub(r"_codebits-\d", "", OPTS.model_path)
        if is_root_node():
            print("loading pretrained model in ", pretrain_path)
        autoencoder.load_pretrain(pretrain_path)
    else:
        scheduler = SimpleScheduler(10)
    if OPTS.resume:
        trainer.load()
    trainer.run()

if OPTS.export_code or OPTS.all:
    from nmtlab.utils import Vocab
    import torch
    assert os.path.exists(OPTS.model_path)
    autoencoder.load(OPTS.model_path)
    out_path = "{}/{}.codes".format(os.getenv("HOME"), os.path.basename(OPTS.model_path).split(".")[0])
    if is_root_node():
        autoencoder.train(False)
        if torch.cuda.is_available():
            autoencoder.cuda()
        c = 0
        c1 = 0
        with open(out_path, "w") as outf:
            print("code path", out_path)
            for batch in dataset.yield_all_batches(batch_size=512):
                src_lines, cfg_lines, src_batch, enc_tree, dec_tree = batch
                out = autoencoder(src_batch.cuda(), enc_tree, dec_tree, return_code=True)
                codes = out["codes"]
                codes_2nd = out["codes_2nd"] if "codes_2nd" in out else None
                for i in range(len(src_lines)):
                    src = src_lines[i]
                    cfg = cfg_lines[i]
                    code = str(codes[i].int().cpu().numpy())
                    if codes_2nd is not None:
                        code = "{} {}".format(code, codes_2nd[i].int().cpu().numpy())
                    outf.write("{}\t{}\t{}\n".format(src, cfg, code))
                outf.flush()
                c += len(src_lines)
                if c - c1 > 10000:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    c1 = c
        sys.stdout.write("\n")

if OPTS.make_target or OPTS.all:
    if is_root_node():
        export_path = "{}/{}.codes".format(os.getenv("HOME"), os.path.basename(OPTS.model_path).split(".")[0])
        out_path = "{}/{}.tgt".format(os.getenv("HOME"), os.path.basename(OPTS.model_path).split(".")[0])
        print("out path", out_path)
        export_map = {}
        for line in open(export_path):
            if len(line.strip().split("\t")) < 3:
                continue
            src, cfg, code = line.strip().rsplit("\t", maxsplit=2)
            code_str = " ".join(["<c{}>".format(int(c) + 1) for c in code.split()])
            export_map["{}\t{}".format(src, cfg)] = code_str
        with open(out_path, "w") as outf:
            src_path = dataset_paths["train_src_path"]
            tgt_path = dataset_paths["train_tgt_path"]
            cfg_path = dataset_paths["train_cfg_path"]
            for src, tgt, cfg in zip(open(src_path), open(tgt_path), open(cfg_path)):
                key = "{}\t{}".format(src.strip(), cfg.strip())
                if key in export_map:
                    outf.write("{} <eoc> {}\n".format(export_map[key], tgt.strip()))
                else:
                    outf.write("\n")

if OPTS.make_oracle_codes:
    if is_root_node():
        from nmtlab.utils import Vocab
        from lib_treedata import TreeDataGenerator
        import torch

        treegen = TreeDataGenerator(dataset_paths["test_cfg_corpus"], dataset_paths["cfg_vocab_path"])
        src_vocab = Vocab(dataset_paths["src_vocab_path"])
        samples = list(zip(open(dataset_paths["test_src_corpus"]), open(dataset_paths["test_cfg_corpus"])))

        print("loading", OPTS.model_path)
        assert os.path.exists(OPTS.model_path)
        autoencoder.load(OPTS.model_path)
        out_path = "{}/{}.test.export".format(os.getenv("HOME"),
                                                                   os.path.basename(OPTS.model_path).split(".")[0])
        autoencoder.train(False)
        if torch.cuda.is_available():
            autoencoder.cuda()
        with open(out_path, "w") as outf:
            print("code path", out_path)
            for i in range(0, len(samples), 512):
                sub_samples = samples[i: i + 512]
                src_lines = [x[0] for x in sub_samples]
                cfg_lines = [x[1] for x in sub_samples]
                processed_samples = []
                for src, cfg in sub_samples:
                    src = src.strip()
                    cfg = cfg.strip()
                    src_ids = src_vocab.encode(src.split())
                    enc_tree, dec_tree = treegen.build_trees(cfg)
                    processed_samples.append((src_ids, enc_tree, dec_tree))
                src_batch, enc_batch, dec_batch = dataset.batch(processed_samples)
                out = autoencoder(src_batch.cuda(), enc_batch, dec_batch, return_code=True)
                codes = out["codes"]
                for j in range(len(src_lines)):
                    src = src_lines[j]
                    cfg = cfg_lines[j]
                    code = codes[j].int()
                    outf.write("{}\n".format(code))
                outf.flush()
                sys.stdout.write(".")
                sys.stdout.flush()
        sys.stdout.write("\n")
