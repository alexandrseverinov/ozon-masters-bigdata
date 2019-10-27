#!/opt/conda/envs/dsenv/bin/python

import os
import sys
import logging

from joblib import load
import pandas as pd


# Init the logger
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

# load the model
model = load("1.joblib")

numeric_features = ["if" + str(i) for i in range(1, 14)]
categorical_features_full = ["cf" + str(i) for i in range(1, 27)] + ["day_number"]
fields_full = ["id"] + numeric_features + categorical_features_full

to_drop_categorical = [20, 1, 22, 10, 21, 12, 11, 23]
categorical_features = (
    [
        "cf" + str(i)
        for i in range(1, 27) if i not in to_drop_categorical
    ] + ["day_number"]
)

fields = ["id"] + numeric_features + categorical_features

# read and inference
read_opts = dict(
        sep="\t", names=fields_full, index_col=False, header=None,
        iterator=True, chunksize=100
)

for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict_proba(df[fields])
    print(
        "\n".join(
            [f"{id},{prediction}"
             for id, prediction in zip(df.id, pred)]
        )
    )
