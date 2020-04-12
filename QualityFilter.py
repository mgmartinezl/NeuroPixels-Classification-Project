from AuxFunctions import *
import os

src = os.path.abspath('C:/Users/NeuroPixels/PycharmProjects/NeuroPixels-Classification/Images/Pipeline/Sample_19-01-16_YC011')
sample = str(src)[-14:]
wanted = {'.csv'}
samples = []

for root, dirs, filenames in os.walk(src, topdown=False):
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext in wanted:
            src_filepath = os.path.join(root, filename)
            df = pd.read_csv(src_filepath, index_col=None, header=0)
            samples.append(df)

df = pd.concat(samples, axis=0, ignore_index=True)
df.to_csv(f'{src}/Summary-Chunks-Sample-{sample}.csv', index=False)

df_units = pd.read_csv(f'{src}/Summary-Chunks-Sample-{sample}.csv')
df_units = df_units.iloc[:, : 14]
df_units = df_units.drop_duplicates()
df_units.to_csv(f'{src}/Summary-Units-Sample-{sample}.csv', index=False)
