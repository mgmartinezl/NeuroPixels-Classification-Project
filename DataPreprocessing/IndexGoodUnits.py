from DataPreprocessing.AuxFunctions import *
import os
import re

data_sets = [
    # "Images/Pipeline/Sample_18-08-30_YC001_probe1",
    # "Images/Pipeline/Sample_18-08-30_YC002_probe1",
    # "Images/Pipeline/Sample_18-08-31_YC003_probe1",
    # "Images/Pipeline/Sample_18-08-31_YC004_probe1",
    # "Images/Pipeline/Sample_18-09-03_YC005_probe1",
    # "Images/Pipeline/Sample_18-09-03_YC006_probe1",
    # "Images/Pipeline/Sample_19-12-13_YC007_probe1",
    # "Images/Pipeline/Sample_18-12-13_YC008_probe1",
    # "Images/Pipeline/Sample_18-12-17_YC009_probe1",
    # "Images/Pipeline/Sample_18-12-17_YC010_probe1",
    # "Images/Pipeline/Sample_18-12-18_YC010_probe1",
    # "Images/Pipeline/Sample_19-01-16_YC011_probe1",
    # "Images/Pipeline/Sample_19-01-23_YC012_probe1",
    # "Images/Pipeline/Sample_19-01-23_YC013_probe1",
    # "Images/Pipeline/Sample_19-01-24_YC012_probe1",
    # "Images/Pipeline/Sample_19-01-24_YC013_probe1",
    # "Images/Pipeline/Sample_19-08-14_YC015_probe1",
    # "Images/Pipeline/Sample_19-08-14_YC015_probe2",
    # "Images/Pipeline/Sample_19-08-14_YC016_probe1",
    # "Images/Pipeline/Sample_19-08-14_YC016_probe2",
    # "Images/Pipeline/Sample_19-08-15_YC016_probe1",
    # "Images/Pipeline/Sample_19-08-16_YC014_probe1",
    # "Images/Pipeline/Sample_19-08-16_YC015_probe1",
    # "Images/Pipeline/Sample_19-09-30_YC017_probe1",
    # "Images/Pipeline/Sample_19-09-30_YC018_probe1",
    # "Images/Pipeline/Sample_19-10-01_YC017_probe1",
    # "Images/Pipeline/Sample_19-10-02_YC019_probe1",
    # "Images/Pipeline/Sample_19-10-02_YC020_probe1",
    # "Images/Pipeline/Sample_19-10-03_YC019_probe1",
    # "Images/Pipeline/Sample_19-10-03_YC020_probe1",
    # "Images/Pipeline/Sample_19-10-22_YC022_probe1",
    # "Images/Pipeline/Sample_19-10-23_YC022_probe1",
    # "Images/Pipeline/Sample_19-11-14_YC029_probe1",
    # "Images/Pipeline/Sample_19-11-14_YC030_probe1",
    # "Images/Pipeline/Sample_19-10-28_YC031_probe1",
    # "Images/Pipeline/Sample_19-11-08_YC038_probe2",
    # "Images/Pipeline/Sample_19-11-05_YC037_probe1",
    # "Images/Pipeline/Sample_19-11-05_YC036_probe1",
    # "Images/Pipeline/Sample_19-11-04_YC037_probe1",
    # "Images/Pipeline/Sample_19-11-04_YC036_probe1",
    # "Images/Pipeline/Sample_19-11-11_YC040_probe1",
    # "Images/Pipeline/Sample_20-15-06_DK186_probe1",
    "Images/Pipeline/Sample_20-27-06_DK187_probe1"

]

df = []

for data_set in data_sets:

    sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", data_set)[0]
    sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", data_set)[0]
    good_path = f'Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files'

    good_units = []

    for root, dirs, filenames in os.walk(good_path, topdown=False):
        for filename in filenames:
            unit = os.path.splitext(filename)[0].split('unit-')[1]
            good_units.append(unit)

            file = os.path.join(root, filename)
            df_unit = pd.read_csv(file)
            good_chunks = df_unit['GoodChunksList'][0]
            good_chunks_wvf = df_unit['WvfChunksList'][0]

            df_ = pd.DataFrame({'Sample': sample_probe,
                                'Unit': unit,
                                'GoodChunks': [good_chunks],
                                'WvfChunks': [good_chunks_wvf]})
            df.append(df_)

    appended_df = pd.concat(df).reset_index(drop=True)
    print(appended_df.head())
    appended_df.to_csv(f'Images/Pipeline/Sample_{sample_probe}/Sample-{sample_probe}-ChunksUnits.csv', index=False)


# for data_set in data_sets:
#
#     sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", data_set)[0]
#     sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", data_set)[0]
#     good_path = f'Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files'
#
#     good_units = []
#
#     for root, dirs, filenames in os.walk(good_path, topdown=False):
#         for filename in filenames:
#             unit = os.path.splitext(filename)[0].split('unit-')[1]
#             good_units.append(unit)
#
#     good_units = [int(i) for i in good_units]
#
#     df_ = pd.DataFrame({'Sample': sample_probe, 'Quality units': len(good_units), 'Units': [good_units]})
#     df.append(df_)
#
# appended_df = pd.concat(df).reset_index(drop=True)
# print(appended_df.head())
# appended_df.to_csv('Images/Pipeline/GoodUnitsAllRecordings.csv', index=False)


# files = []
#
# for data_set in data_sets:
#
#     sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", data_set)[0]
#     sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", data_set)[0]
#     good_path = f'Images/Pipeline/Sample_{sample_probe}/Features/Files'
#
#     for root, dirs, filenames in os.walk(good_path, topdown=False):
#         for filename in filenames:
#             file = os.path.join(root, filename)
#             files.append(file)
#
#     print(f"Overall sample now contains {len(files)} observations")
#
#     combined_csv = pd.concat([pd.read_csv(f) for f in files])
#     combined_csv.to_csv("Neuropixels_Features.csv", index=False)

# files = []
# good_path = 'C:/Users/NeuroPixels/PycharmProjects/NeuroPixels-Classification/DataAnalysis/ACGs'
#
# for root, dirs, filenames in os.walk(good_path, topdown=False):
#     print('Hello')
#     for filename in filenames:
#         file = os.path.join(root, filename)
#         files.append(file)
#
#     print(f"Overall sample now contains {len(files)} observations")
#
# print(len(files))
# combined_csv = pd.concat([pd.read_csv(f) for f in files])
# print(combined_csv)
# combined_csv.to_csv("../DataAnalysis/Neuropixels_ACGs.csv", index=False)

