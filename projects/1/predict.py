#!/opt/conda/envs/dsenv/bin/python

import os
import sys
import logging

from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields


# Init the logger
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

# load the model
model = load("1.joblib")

# read and inference
read_opts = dict(
        sep="\t", names=fields, index_col=False, header=None,
        iterator=True, chunksize=100
)

for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict_proba(df)
    print(
        "\n".join(
            [f"{id},{prediction}"
             for id, prediction in zip(df.id, pred)]
        )
    )
