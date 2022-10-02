from pathlib import Path

import pandas as pd
import yaml
from pandas import json_normalize
from tqdm import tqdm


def collect_results() -> pd.DataFrame:
    results = pd.DataFrame()

    outputs = Path("outputs")
    multirun = Path("multirun")

    for day in tqdm(list(outputs.iterdir()), desc="Processing days (outputs folder)", leave=False):
        for experiment in tqdm(list(day.iterdir()), desc="Processing experiments", leave=False):
            results = process_experiment(experiment, results)

    for day in tqdm(multirun.iterdir(), desc="Processing days (multirun folder)", leave=False):
        for time in tqdm(day.iterdir(), desc="Processing launch times", leave=False):
            for experiment in tqdm(time.iterdir(), desc="Processing experiments", leave=False):
                if experiment.is_dir():
                    results = process_experiment(experiment, results)

    return results


def process_experiment(experiment: Path, results: pd.DataFrame):
    try:
        score = (experiment / "score.txt").read_text()
        if score == "nan":
            return results
        try:
            score = float(score)
        except ValueError:
            print(score)
            pass

        config = (experiment / ".hydra" / "config.yaml").read_text()
        config = yaml.safe_load(config)

        row = pd.concat([
            json_normalize(config), 
            pd.DataFrame([{"score": score}])
        ], axis=1)
        return pd.concat([results, row])

    except FileNotFoundError:
        return results
        
if __name__ == "__main__":
    results = collect_results()
    results.drop_duplicates(inplace=True)
    results.to_csv("results.csv", sep="\t", index=False)
    results.to_excel("results.xlsx", index=False)



    