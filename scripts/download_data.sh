mkdir data
gdown "1hpNYy84T8vt74JgtqpNlkZYnQdXhWBHI" -O data/data.zip
gdown "1wEcg0r-HaA2RuWQBWuR2DJiaK7OA1-Si" -O data/csv_folds.zip

mkdir data/csvs

unzip data/data.zip -d data
unzip data/csv_folds.zip -d data/csvs

mv data/Train data/raw