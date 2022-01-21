import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import argparse

parser = argparse.ArgumentParser('Training Object Detection')
parser.add_argument('-k','--num_folds', type=int, default=5, help='Number of k-folds')
parser.add_argument('--csv_file', type=str, help='csv path file')
parser.add_argument('--task', type=int, default=1, help='task nubmer')

def get_new_labels(y):
    y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_new

ROOT = './splits'

if __name__ == '__main__':
    args = parser.parse_args()
    skf = StratifiedKFold(n_splits=args.num_folds)

    df = pd.read_csv(args.csv_file)

    X=df.id

    if args.task == 1:
        y=df.iloc[:, 1]
    if args.task == 2:
        y=df.iloc[:, 2:9]
    if args.task == 3:
        y=df.iloc[:, 9:]

    y = y.to_numpy()

    y_new = get_new_labels(y)

    for fold_id, (train_index, test_index) in enumerate(skf.split(X, y_new)):
        train_df = df.iloc[train_index]
        val_df = df.iloc[test_index]

        train_df.to_csv(f"{ROOT}/task{args.task}/task{args.task}_train_fold{fold_id}.csv", index=False)
        val_df.to_csv(f"{ROOT}/task{args.task}/task{args.task}_val_fold{fold_id}.csv", index=False)

    