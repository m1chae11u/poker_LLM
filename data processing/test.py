import pandas as pd


if __name__ == "__main__":
    CSV_FILENAME = "postflop_500k_train_set_25252525.csv"
    dataset = pd.read_csv(CSV_FILENAME).fillna("")
    print(dataset)
    print(f"\n List of all columns in dataset: {dataset.columns}")