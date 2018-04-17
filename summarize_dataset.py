from pathlib import Path
from tqdm import tqdm
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-n", default=False)
args = parser.parse_args()


out_path = Path("D:/Projects/Lerot3/out")
experiment_name = args.n if args.n else "ga_pm_test"
um = [
    "inf",
    "nav",
    "per"
]
datasets = [  
        "OHSUMED",
        # "NP2003",
        # "HP2003",
        # "HP2004",
        # "MQ2007",
        # "MQ2008",
        # "NP2004",
        # "TD2003",
        # "TD2004"
        ]
folds = ["Fold" + str(i) for i in range(1, 6)]
runs = [f"{i:03d}" for i in range(0, 1)]
filename = "metrics.csv"

def get_discount(num_queries):
    gamma = 1 - 5 / num_queries
    gm = pd.Series([gamma ** i for i in range(0, num_queries)])
    return gm


for cm in tqdm(um, desc='cm'):
    for ds in tqdm(datasets, desc='dataset'):
        online_df = pd.DataFrame()
        offline_df = pd.DataFrame()
        for fold in tqdm(folds, desc='folds'):
            for run in tqdm(runs, desc='runs'):
                fn = out_path / experiment_name / cm / ds / fold / run / filename
                df = pd.read_csv(str(fn))
                if not online_df.empty:
                    online_df = pd.concat((online_df, df.loc[:, [col for col in df.columns if col.endswith("_online")]]), axis=1)
                else:
                    online_df = df.loc[:, :]
                if not offline_df.empty:
                    offline_df = pd.concat((offline_df, df.loc[:, [col for col in df.columns if col.endswith("_offline")]]), axis=1)
                else:
                    offline_df = df.loc[:, :]
        disc = get_discount(len(online_df))
        online_df = online_df.multiply(disc, axis='index').cumsum()
        online_perf = online_df.mean(axis=1)
        online_std = online_df.std(axis=1)
        offline_perf = offline_df.mean(axis=1)
        offline_std = offline_df.std(axis=1)
        perf = pd.concat((online_perf, online_std, offline_perf, offline_std), axis=1)
        perf.columns = ["online_mean@10", "online_std@10", "offline_mean@10", "offline_std@10"]

        if (out_path / experiment_name / cm / ds / "aggregated.csv").exists():
            raise Exception("EXPERIMENT ALREADY CONDUCTED")

        perf.to_csv(str(out_path / experiment_name / cm / ds / "aggregated.csv"), index=False, header=True)
