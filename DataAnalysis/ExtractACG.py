"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: QualityCheckPipeline.py
Description: implementation of quality metrics on units and chunks to choose the right ones for the next stage in the
classification pipeline.

"""

from DataPreprocessing.AuxFunctions import *
import pandas as pd
from operator import itemgetter
from rtn.npix.spk_t import trn
from scipy import interpolate
from scipy.interpolate import interp1d
import os
import re
import ast


np.seterr(all='ignore')

data_sets = [
    'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1',
    'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',
    'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1',
    'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',
    'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1',
    'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2'
]

# Parameters
fs = 30000
c_bin = 0.5
c_win = 80
violations_ms = 0.8
exclusion_quantile = 0.02
unit_size_s = 20 * 60
unit_size_ms = unit_size_s * 1000
chunk_size_s = 60
chunk_size_ms = 60 * 1000
samples_fr = unit_size_s * fs
n_chunks = int(unit_size_s / chunk_size_s)
n_waveforms = 400
ignore_nwvf = False
t_waveforms = 120
again = False

samples = []
units = []
acgs = []
acgs_x = []


for dp in data_sets:

    df_all = []

    sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", dp)[0]
    sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", dp)[0]
    print('***************************')
    print('Sample:', sample_probe)
    cell_type = dp[8:11]

    # Load kilosort aux files
    amplitudes_sample = np.load(f'{dp}//amplitudes.npy')  # shape N_tot_spikes x 1
    spike_times = np.load(f'{dp}//spike_times.npy')  # in samples
    spike_clusters = np.load(f'{dp}//spike_clusters.npy')

    # Extract good units of current sample
    all_good_units = pd.read_csv('../Images/Pipeline/GoodUnitsAllRecordings.csv')
    good_units = ast.literal_eval(all_good_units[all_good_units['Sample'].str.match(sample_probe)]['Units'].values[0])

    # paths = [f'../Images/Pipeline/Sample_{sample_probe}/Features/Images',
    #          f'../Images/Pipeline/Sample_{sample_probe}/Features/Files']

    # print(f"Quality units found in current sample: {len(good_units)} --> {good_units}")

    # Extract chunks for good units
    units_chunks = pd.read_csv(f'../Images/Pipeline/Sample_{sample_probe}/Sample-{sample_probe}-ChunksUnits.csv')

    # good_units = [232]

    for unit in good_units:

        # for features_path in paths:
        #     if not os.path.exists(features_path):
        #         os.makedirs(features_path)

        print(f'Unit {good_units.index(unit)+1} of {len(good_units)}')
        print(f'Unit {unit}')

        try:
            # We only care about extracting the ACG
            ACG = acg(dp, unit, c_bin, c_win, subset_selection=[(0, unit_size_s)])
            x_unit = np.linspace(-c_win * 1. / 2, c_win * 1. / 2, ACG.shape[0])
            y_unit = ACG.copy()
            y_unit = y_unit / max(y_unit)
            y_lim1_unit = 0
            yl_unit = max(y_unit)
            y_lim2_unit = int(yl_unit) + 2 - (yl_unit % 5)

            # Save df
            # **********************************************************************************************************
            # samples.append(sample_probe)
            # units.append(unit)
            # acgs.append(y_unit)
            # acgs_x.append(x_unit)

            column_names = ['ACG_' + str(i) for i in range(0, len(y_unit))]

            df1 = [y_unit]
            df = pd.DataFrame(data=df1, columns=column_names)
            df['Sample'] = [sample_probe]
            df['Unit'] = [unit]
            df_all.append(df)

            # Plot
            # ***********************************************************************************************************
            # fig, axs = plt.subplots(1, 1, figsize=(10, 7))
            #
            # plt.bar(x=x_unit, height=y_unit, width=0.2, color='salmon', bottom=y_lim1_unit)
            # axs.set_ylim([y_lim1_unit, y_lim2_unit])
            # axs.set_ylabel("Autocorrelation (Hz)", size=9, color='#939799')
            # axs.set_xlabel("Time (ms)", size=9)
            # axs.set_title(f' \n \n Autocorrelogram', fontsize=9, loc='center', color='#939799')
            #
            # format_plot(axs)
            # fig.tight_layout()
            # plt.show()

        except Exception:
            continue

    appended_df = pd.concat(df_all).reset_index(drop=True)
    print(appended_df.head(5))

    appended_df.to_csv(f'ACGs/ACGs_{sample_probe}.csv', index=False)
    print(f'ACGs_{sample_probe}.csv  ---> Successfully created!')

