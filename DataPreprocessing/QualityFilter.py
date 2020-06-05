import os
import re

data_sets = [
    # 'C:/routinesMem/18-08-30_YC001_probe1',
    # 'C:/routinesMem/18-08-30_YC002_probe1',
    # 'C:/routinesMem/18-08-31_YC003_probe1',
    # 'C:/routinesMem/18-08-31_YC004_probe1',
    # 'D:/routinesMem/18-09-03_YC006_probe1',
    # 'C:/routinesMem/18-12-17_YC009_probe1',
    # 'C:/routinesMem/18-12-18_YC010_probe1',
    # 'C:/routinesMem/19-01-16_YC011_probe1',
    # 'C:/routinesMem/19-08-14_YC015_probe1',
    # 'C:/routinesMem/19-08-14_YC015_probe2',
    # 'C:/routinesMem/19-08-14_YC016_probe2',
    # 'C:/routinesMem/19-08-16_YC014_probe1',
    # 'C:/routinesMem/19-11-11_YC040_probe2',
    # 'C:/routinesMem/18-09-03_YC005_probe1',
    # 'C:/routinesMem/19-10-02_YC019_probe1',
    # 'C:/routinesMem/19-01-24_YC012_probe1',
    # 'C:/routinesMem/19-10-02_YC020_probe1',
    # 'C:/routinesMem/19-10-03_YC020_probe1',
    # 'C:/routinesMem/19-10-03_YC019_probe1',
    # 'C:/routinesMem/19-01-23_YC012_probe1',
    # 'C:/routinesMem/19-01-23_YC013_probe1',
    # 'C:/routinesMem/19-01-23_YC013_probe1',
    # 'C:/routinesMem/19-09-30_YC017_probe1',
    # 'C:/routinesMem/19-09-30_YC018_probe1',
    # 'C:/routinesMem/19-10-01_YC017_probe1',
    # 'C:/routinesMem/19-11-14_YC029_probe1',
    # 'C:/routinesMem/19-11-14_YC030_probe1',

    # Meta
    # 'C:/routinesMem/18-12-17_YC010_probe1',
    # 'C:/routinesMem/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'D:/routinesMem/19-08-16_YC015_probe1',

    # 'C:/routinesMem/19-01-24_YC013_probe1',
    # 'C:/routinesMem/19-12-13_YC007_probe1',

    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',

    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2'
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2'
    # 'Z:/npix_data/optotagging/Back-Up-YYC_2019_optotagging/data/GrC/19-10-23_YC022/19-10-23_YC022_probe1'
    # 'Z:/npix_data/optotagging/Back-Up-YYC_2019_optotagging/data/GrC/19-10-22_YC022/19-10-22_YC022_probe1'
    # 'Z:/npix_data/optotagging/Back-Up-YYC_2019_optotagging/data/GrC/19-11-05_YC036/19-11-05_YC036_probe1'  # Not finished yet
    # 'Z:/npix_data/optotagging/Back-Up-YYC_2019_optotagging/data/GrC/19-11-04_YC037/19-11-04_YC037_probe1'  # Not finished yet
    # 'Z:/npix_data/optotagging/Back-Up-YYC_2019_optotagging/data/GrC/19-11-04_YC036/19-11-04_YC036_probe1'  # Not finished yet

    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    'F:/data/GrC/19-11-05_YC036/19-11-05_YC036_probe1'


]

# src = f"Images/Pipeline/Sample_19-01-16_YC011/Pipeline2/BadUnits"
# sample = str(src)[-14:]
# wanted = {'.csv'}
# samples = []

dict_good_units = {}

for data_set in data_sets:

    sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", data_set)[0]
    sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", data_set)[0]
    good_path = f'Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files'
    routines_path = f'{data_set}/routinesMemory'
    print(routines_path)

    good_units = []

    for root, dirs, filenames in os.walk(good_path, topdown=False):
        for filename in filenames:
            unit = os.path.splitext(filename)[0].split('unit-')[1]
            good_units.append(unit)

    good_units = [int(i) for i in good_units]
    print('Good units in this sample:', len(good_units))
    print(good_units)

    dict_good_units.update({sample_probe: good_units})
    print(dict_good_units)

    for root, dirs, filenames in os.walk(routines_path, topdown=False):
        for filename in filenames:
            unit_pattern = re.findall(r"\w{3}\[?\d{1,4}\]?", filename)
            unit_pattern = int(re.findall(r"\d+", unit_pattern[0])[0])
            if unit_pattern not in good_units:
                try:
                    src_filepath = os.path.join(root, filename)
                    if os.path.isfile(src_filepath) or os.path.islink(src_filepath):
                        os.unlink(src_filepath)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (filename, e))



    # for root, dirs, filenames in os.walk(src, topdown=False):
    #     for filename in filenames:
    #         ext = os.path.splitext(filename)[1]
    #         if ext in wanted:
    #             src_filepath = os.path.join(root, filename)
    #             df = pd.read_csv(src_filepath, index_col=None, header=0)
    #             samples.append(df)

# df = pd.concat(samples, axis=0, ignore_index=True)
# df.Unit = pd.to_numeric(df.Unit, errors='coerce')
# df = df.sort_values('Unit')
# df.to_csv(f'{src}/Bad-Units-Sample-19-01-16_YC011.csv', index=False)

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