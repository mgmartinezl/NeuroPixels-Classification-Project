from AuxFunctions import *
import os

src = f"Images/Pipeline/Sample_19-01-16_YC011/Pipeline2/BadUnits"
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
df.Unit = pd.to_numeric(df.Unit, errors='coerce')
df = df.sort_values('Unit')
df.to_csv(f'{src}/Bad-Units-Sample-19-01-16_YC011.csv', index=False)

# df_units = pd.read_csv(f'{src}/Summary-Chunks-Sample-{sample}.csv')
# df_units = df_units.iloc[:, : 18]
# df_units = df_units.drop_duplicates()
# df_units.to_csv(f'{src}/Summary-Units-Sample-{sample}.csv', index=False)
#
# df_chunks = pd.read_csv(f'{src}/Summary-Chunks-Sample-{sample}.csv')
# df_filtered = df_chunks[(df_chunks['Unit_Spikes'] >= 300) &
#                         (df_chunks['Unit_Fp'] <= 5) &
#                         (df_chunks['Chunk_Wvf_Peaks'] <= 3) &
#                         (df_chunks['Chunk_Biggest_Peak'] == 1) &
#                         (df_chunks['Chunk_Missing_Spikes'] <= 30)]
#
# df_filtered.to_csv(f'{src}/Good-Units-Chunks-Sample-{sample}.csv', index=False)