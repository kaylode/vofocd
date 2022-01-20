import argparse
import random
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='raw.csv',
                    help='path to csv file')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--out', type=str, default='.',
                    help='directory to save the splits (default: .)')
parser.add_argument('--n_splits', type=int, default=5,
                    help='Number of folds, at least 2')
parser.add_argument('--shuffle', type=bool, default=True,
                    help='Whether to shuffle each class’s samples before splitting into batches (default: True)')

if __name__ == '__main__':
    args = parser.parse_args()

    # Seed the random processes
    random.seed(args.seed)

    # Load CSV
    df = pd.read_csv(args.csv)

    X_train = df.iloc[:, :-1]
    Y_train = df.iloc[:, -1]

    cv = StratifiedKFold(n_splits=args.n_splits,
                         random_state=args.seed, shuffle=args.shuffle)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, Y_train)):

        TRAIN = str(fold) + '_train.csv'
        VAL = str(fold) + '_val.csv'

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        if not os.path.exists(os.path.join(args.out, str(fold))):
            os.makedirs(os.path.join(args.out, str(fold)))

        train_csv_path = os.path.join(args.out, str(fold), TRAIN)
        val_csv_path = os.path.join(args.out, str(fold), VAL)

        train_df.to_csv(train_csv_path, encoding='utf8', index=False)
        val_df.to_csv(val_csv_path, encoding='utf8', index=False)
