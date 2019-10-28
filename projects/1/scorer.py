#!/opt/conda/envs/dsenv/bin/python

# This is a LogLoss scorer

import sys
import os
from glob import glob
import logging
import math


# Init the logger
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

score = 0
n_records = 0

prev_key = None
values = []


def logloss(true_label, predicted, eps=1e-15):
    p = max(eps, min(predicted, 1 - eps))
    if true_label == 1:
        return p
    return 1 - p


for line in sys.stdin:
    key, value = line.strip().split(",")

    if key != prev_key and prev_key is not None:
        score += logloss(values[0], values[1])
        n_records += 1
        values = []
    values.append(float(value))
    prev_key = key

if prev_key is not None:
    score += logloss(values[0], values[1])
    n_records += 1

score = -math.log(score)

print(score)

sys.exit(0)
