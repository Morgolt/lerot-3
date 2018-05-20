import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_df(online_mean, online_std, offline_mean, offline_std, cm, ds):
    print(f"ds: {ds}, cv: {cm}, counts: {len(online_mean)}, {len(online_std)}, {len(offline_mean)}, {len(offline_std)}")
    df = pd.DataFrame()
    df['online_mean@10'] = online_mean
    df['online_std@10'] = online_std
    df['offline_mean@10'] = offline_mean
    df['offline_std@10'] = offline_std
    # df['click_model'] = cm
    # df['dataset'] = ds
    return df


def get_offline(off, step=10, num_queries=10001):
    out = np.zeros(num_queries)
    out[::step] = off[:len(out[::step])]
    return fill_zeros_with_last(out)


def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


parser = argparse.ArgumentParser()
parser.add_argument("-n", default=False)
args = parser.parse_args()

experiment = args.n
MODEL = 'PMGD19cand'

um = {
    "perfect": "per",
    "navigational": "nav",
    "informational": "inf"
}
datasets = {
    "OHSUMED": "OHSUMED",
    "2003_np": "NP2003",
    "2003_hp": "HP2003",
    "2004_hp": "HP2004",
    "MQ2007": "MQ2007",
    "MQ2008": "MQ2008",
    "2004_np": "NP2004",
    "2003_td": "TD2003",
    "2004_td": "TD2004"
}

DATA_PATH = Path(f"/mnt/c/Users/Rodion_Martynov/Documents/projects/Lerot3/out/{experiment}/average")

results = {}

for ds_key, ds in datasets.items():
    path = DATA_PATH / ds_key
    out = DATA_PATH / ".." / ds
    results = {}
    with open(DATA_PATH / ds_key / MODEL, 'r') as fn:
        found = False
        while not found:
            line = fn.readline()
            first = line.split(' ')[0]
            if first in um.keys():
                found = True
        perf_test_mean = list(map(float, fn.readline().split(' ')[:-1]))
        perf_test_mean = get_offline(perf_test_mean)
        fn.readline()

        perf_test_std = list(map(float, fn.readline().split(' ')[:-1]))
        perf_test_std = get_offline(perf_test_std)
        fn.readline()

        nav_test_mean = list(map(float, fn.readline().split(' ')[:-1]))
        nav_test_mean = get_offline(nav_test_mean)
        fn.readline()

        nav_test_std = list(map(float, fn.readline().split(' ')[:-1]))
        nav_test_std = get_offline(nav_test_std)
        fn.readline()

        inf_test_mean = list(map(float, fn.readline().split(' ')[:-1]))
        inf_test_mean = get_offline(inf_test_mean)
        fn.readline()

        inf_test_std = list(map(float, fn.readline().split(' ')[:-1]))
        inf_test_std = get_offline(inf_test_std)
        found = False

        while not found:
            line = fn.readline()
            if line == 'perfect TRAIN ONLINE MEAN 0.9995\n':
                found = True
        perf_train_mean = list(map(float, fn.readline().split(' ')[:-1]))
        fn.readline()
        perf_train_std = list(map(float, fn.readline().split(' ')[:-1]))
        found = False

        while not found:
            line = fn.readline()
            if line == 'navigational TRAIN ONLINE MEAN 0.9995\n':
                found = True

        nav_train_mean = list(map(float, fn.readline().split(' ')[:-1]))
        fn.readline()
        nav_train_std = list(map(float, fn.readline().split(' ')[:-1]))
        fn.readline()
        found = False

        while not found:
            line = fn.readline()
            if line == 'informational TRAIN ONLINE MEAN 0.9995\n':
                found = True

        inf_train_mean = list(map(float, fn.readline().split(' ')[:-1]))
        fn.readline()
        inf_train_std = list(map(float, fn.readline().split(' ')[:-1]))
        fn.readline()

        results['per'] = make_df(perf_train_mean, perf_train_std, perf_test_mean, perf_test_std, 'per', ds)
        results['nav'] = make_df(nav_train_mean, nav_train_std, nav_test_mean, nav_test_std, 'nav', ds)
        results['inf'] = make_df(inf_train_mean, inf_train_std, inf_test_mean, inf_test_std, 'inf', ds)

        for cm in um.values():
            out = DATA_PATH / '..' / cm / ds
            if not out.exists():
                out.mkdir(parents=True)
            results[cm].to_csv(str(out / 'aggregated.csv'), header=True, index=False)
