#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re


def build_cfg_map(cfg_path, oneline_cfg=False):
    cfg_lines = []
    if oneline_cfg:
        cfg_lines = list(map(str.strip, open(cfg_path)))
    else:
        buf = []
        for line in open(cfg_path):
            line = line.strip()
            if not line:
                if buf:
                    cfg_lines.append(" ".join(buf))
                    buf.clear()
                else:
                    pass
            else:
                buf.append(line)
    # building CFG map
    cfg_map = {}
    for cfg_line in cfg_lines:
        words = re.findall(r" ([^\) ]+)\)", cfg_line)
        sent = "".join(words)
        sent = sent.replace("-LRB-", "(")
        sent = sent.replace("-RRB-", ")")
        sent = sent.replace("-LSB-", "[")
        sent = sent.replace("-RSB-", "]")
        sent = sent.replace("`", "'")
        sent = sent.replace("’", "'")
        sent = sent.replace(".", "")
        cfg_map[sent] = cfg_line
    return cfg_map


def align_cfg_oneline(sent_path, cfg_path, oneline_cfg=False):
    sents = map(str.strip, open(sent_path))
    cfg_map = build_cfg_map(cfg_path, oneline_cfg=True)
    cfg_onelines = []
    fails = 0
    for sent in sents:
        key = sent.replace(" ", "")
        key = key.replace("&apos;", "'")
        key = key.replace("&quot;", "''")
        key = key.replace("\"", "''")
        key = key.replace("’", "'")
        key = key.replace(".", "")
        key = key.replace("—", "--")
        if key in cfg_map:
            cfg_onelines.append(cfg_map[key])
        else:
            cfg_onelines.append("")
            fails += 1
    print("fails", fails)
    return cfg_onelines


