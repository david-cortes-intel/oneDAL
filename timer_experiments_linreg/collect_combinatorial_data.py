import numpy as np, polars as pl
from math import ceil
import subprocess
import itertools
import re

m_try = [8, 16, 32, 64, 128, 256, 512, 1024, 5_000, 10_000, 50_000]
n_try = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
batch_size_try = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 50_000]
nthreads_try = [1, 2, 8]
num_repeats = 6


combs = itertools.product(m_try, n_try, batch_size_try, nthreads_try)

results_sep_process = list()
for comb in combs:
    m, n, batch_size, nthreads = comb
    if batch_size > m:
        batch_size = m
    if ceil(m / batch_size) < nthreads:
        continue

    ## With the data being re-generated every time, but in the same process
    res = subprocess.run(
        [
            "python",
            "timer_simple.py",
            f"m={m}",
            f"n={n}",
            f"batch_size={batch_size}",
            f"nthreads={nthreads}",
            f"repeated_calls_full={num_repeats}",
            "repeated_calls_fit=1",
            "same_seed=1",
            "method=uniform",
        ],
        stdout=subprocess.PIPE,
    ).stdout.decode().splitlines()
    assert res[0].startswith("time gen:")
    assert res[-1].startswith("time fit:")
    entry_base = {
        "m": m,
        "n": n,
        "nthreads": nthreads,
        "sampling": "uniform",
        "method": "same process, re-generating data",
    }
    entry = None
    repetition_num = 0
    for ln in res:
        section, value = ln.split(":")
        section = re.sub("time", "", section).strip()
        value = float(value.strip())
        if section == "gen":
            repetition_num += 1
            entry = entry_base | {"repetition": repetition_num}

        assert isinstance(entry, dict)
        entry[section] = value
        if section == "fit":
            results_sep_process.append(entry)
            entry = None


# with pl.Config() as cfg:
#     cfg.set_tbl_cols(20)
#     print(
#         pl.DataFrame(results_sep_process)
#     )

(
    pl.DataFrame(results_sep_process, infer_schema_length=None, strict=False)
    .write_parquet("data_combinatorial.parquet")
)

import pickle
with open("data_combinatorial.pkl", "wb") as f:
    pickle.dump(results_sep_process, f)
