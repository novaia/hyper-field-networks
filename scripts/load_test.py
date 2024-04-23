import datasets

dataset = datasets.load_dataset(
    'parquet', 
    data_files={'train': 'data/colored_monsters_encoded_alt_fixed/0762.parquet'}, 
    split='train'
)
