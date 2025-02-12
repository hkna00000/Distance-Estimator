'''
Purpose: Generate dataset for depth estimation
'''
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

df = pd.read_csv('annotations1.csv')
new_df = df.loc[df['class'] != 'DontCare']
result_df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', \
                           'angle', 'xloc', 'yloc', 'zloc'])

pbar = tqdm(total=new_df.shape[0], position=1)

for idx, row in new_df.iterrows():
    pbar.update(1)
    file_path = os.path.exists(os.path.join("original_data/train_annots", row['filename']))
    if os.path.exists(file_path):
        result_df.at[idx, 'filename'] = row['filename']

        try:
            result_df.at[idx, 'xmin'] = int(row['xmin'])
            result_df.at[idx, 'ymin'] = int(row['ymin'])
            result_df.at[idx, 'xmax'] = int(row['xmax'])
            result_df.at[idx, 'ymax'] = int(row['ymax'])
            result_df.at[idx, 'xloc'] = int(row['xloc'])
            result_df.at[idx, 'yloc'] = int(row['yloc'])
            result_df.at[idx, 'zloc'] = int(row['zloc'])
        except ValueError as e:
            print(f"ValueError at row {idx}: {e}")
            continue

        result_df.at[idx, 'angle'] = row['observation angle']
    else:
        print(f"File not found: {file_path}")

mask = np.random.rand(len(result_df)) < 0.9
train = result_df[mask]
test = result_df[~mask]

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
