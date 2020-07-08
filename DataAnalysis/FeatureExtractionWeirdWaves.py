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

    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1'  # DONE
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1'  # DONE
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1'  # DONE
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1'  # DONE
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1'  # DONE
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1'  # DONE
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1'  # DONE
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1'  # DONE
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1'  # DONE
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1'  # DONE
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1'  # DONE
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1'  # DONE
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1'  # DONE
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1'  # DONE
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1'  # DONE
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2'  # DONE
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1'  # DONE
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1'  # DONE
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1'  # DONE
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1'  # DONE
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1'  # DONE
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1'  # DONE
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1'  # DONE
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1'  # DONE
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1'  # DONE
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1'  # DONE
    # 'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1'  # DONE
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1'  # DONE
    # 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1'  # DONE
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1'  # DONE
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1'  # DONE
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1'  # DONE
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1'  # DONE 359, 407 ERRORS
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2'  # DONE
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe2'  # NO GOOD UNITS
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1'   # DONE
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2'  # 344, 373, 591, 613, 62 DONE
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1'  # DONE
    # DK 186 and 187 missing for Golgi
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2'  # 344, 373, 591, 613, 62
    'D:/Recordings/20-27-06_DK187/20-27-06_DK187_probe1'
]

# Parameters
fs = 30000
exclusion_quantile = 0.02
unit_size_s = 20 * 60
unit_size_ms = unit_size_s * 1000
chunk_size_s = 60
chunk_size_ms = 60 * 1000
samples_fr = unit_size_s * fs
n_chunks = int(unit_size_s / chunk_size_s)
n_waveforms = 400
ignore_nwvf = True
t_waveforms = 120
again = False

for dp in data_sets:

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

    paths = [f'../Images/Pipeline/Sample_{sample_probe}/Features/Images',
             f'../Images/Pipeline/Sample_{sample_probe}/Features/Files']

    print(f"Quality units found in current sample: {len(good_units)} --> {good_units}")

    # Extract chunks for good units
    units_chunks = pd.read_csv(f'../Images/Pipeline/Sample_{sample_probe}/Sample-{sample_probe}-ChunksUnits.csv')

    good_units = [265]

    d = 3
    h = 2
    r = 10

    for unit in good_units:

        for features_path in paths:
            if not os.path.exists(features_path):
                os.makedirs(features_path)

        print(f'Unit >>>> {unit}')

        # Extract chunks
        chunks = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['GoodChunks'][
            units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[0]])))

        chunks_wvf = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['WvfChunks'][
            units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[0]])))

        # chunks_wvf = [3, 4]

        # Store quality metrics
        unit_quality_metrics = pd.read_csv(f'../Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files/Sample-{sample}-unit-{unit}.csv')

        # Retrieve peak channel for this unit
        peak_channel = unit_quality_metrics['PeakChUnit'][0]

        # Collect for later...
        l_chunk_trns = []
        l_chunk_wvfs = []

        # Read chunks...
        for i in range(n_chunks):
            if i in chunks:

                # These chunks will be used to compute temporal features
                # Chunk length and times in seconds
                chunk_start_time = i * chunk_size_s
                chunk_end_time = (i + 1) * chunk_size_s

                # Chunk spikes
                trn_samples_chunk = trn(dp, unit=unit,
                                        subset_selection=[(chunk_start_time, chunk_end_time)],
                                        enforced_rp=0.5, again=again)

                l_chunk_trns.append(trn_samples_chunk)

            else:
                continue

        for i in range(n_chunks):
            if i in chunks_wvf:

                # These chunks will be used to compute waveform-based features. They will be maximum 3
                # Chunk length and times in seconds
                chunk_start_time = i * chunk_size_s
                chunk_end_time = (i + 1) * chunk_size_s

                # Chunk mean waveform around peak channel
                chunk_wvf = np.mean(
                    wvf(dp, unit,
                        n_waveforms=n_waveforms,
                        t_waveforms=t_waveforms,
                        subset_selection=[(chunk_start_time, chunk_end_time)],
                        again=again,
                        ignore_nwvf=ignore_nwvf)
                    [:, :, peak_channel], axis=0)
                l_chunk_wvfs.append(chunk_wvf)

        # *********************************************************************************************************

        # Put chunks together
        # Join trn samples of good chunks for temporal features computation
        trn_samples_block = np.concatenate(l_chunk_trns).ravel()

        # Compute Inter-Spike Interval for the block of good chunks
        # We need to exclude the times between chunks with the exclusion quartile
        isi_block_clipped = compute_isi(trn_samples_block, quantile=exclusion_quantile)
        isi_block_no_clipped = compute_isi(trn_samples_block)

        # Estimate number of optimal bins for ISI
        isi_bins = estimate_bins(isi_block_no_clipped, rule='Sqrt')

        isi_log_bins = None

        try:
            isi_log_bins = np.geomspace(isi_block_no_clipped.min(), isi_block_no_clipped.max(), isi_bins)
        except ValueError:
            pass

        # Compute temporal features for the unit
        mfr = mean_firing_rate(isi_block_clipped)
        mifr, med_isi, mode_isi, prct5ISI, entropy, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, skw \
            = compute_isi_features(isi_block_clipped)

        # Plot
        # ***********************************************************************************************************
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))

        # Plot temporal features
        axs[0].hist(isi_block_no_clipped, bins=isi_log_bins, color='lightgray')
        axs[0].set_xscale('log')
        # axs[1, 0].set_xlim([0.1, 1000])
        axs[0].set_xlim(left=0.1)
        axs[0].set_xlabel('ms (log scale)', size=9)
        # axs[0].set_xlabel('ms', size=9)
        axs[0].set_title(f' \n \n \n Inter Spike Interval  \n Chunks {chunks}',
                         fontsize=9, loc='center', color='#939799')

        leg_temp_features = [f'mfr = {mfr} Hz \n'
                             f'mifr = {mifr} Hz \n'
                             f'Entropy = {entropy} \n'
                             f'Median (isi) = {med_isi}  \n'
                             f'Mode (isi) = {mode_isi}  \n'
                             f'Perc 5th (isi) = {prct5ISI}  \n'
                             f'Mean CV2 = {CV2_mean}  \n'
                             f'Median CV2 = {CV2_median}  \n'
                             f'Mean CV = {CV}  \n'
                             f'IR = {IR}  \n'
                             f'Lv = {Lv}  \n'
                             f'LvR = {LvR}  \n'
                             f'LcV = {LcV}  \n'
                             f'SI = {SI}  \n'
                             f'skw = {skw}  \n']

        leg_0 = axs[0].legend(leg_temp_features, loc='best', frameon=False, fontsize=7)

        for text in leg_0.get_texts():
            text.set_color("gray")

        for lh in leg_0.legendHandles:
            lh.set_alpha(0)

        plt.suptitle(f'Recording {sample_probe} - Unit {unit}', fontsize=10, color='#939799')
        format_plot(axs[0])
        format_plot(axs[1])
        plt.tight_layout()
        plt.show()

        fig.savefig(f'{paths[0]}/Sample-{sample}-unit-{unit}.png')
        print(f'{paths[0]}/Sample-{sample}-unit-{unit}.png file successfully saved!')

        unit_temporal_features = pd.DataFrame({
                                    'Unit': unit,
                                    'tf_MIFRBlockHz': mifr,
                                    'tf_MedIsi': med_isi,
                                    'tf_ModeIsi': mode_isi,
                                    'tf_Perc5Isi': prct5ISI,
                                    'tf_Entropy': entropy,
                                    'tf_CV2Mean': CV2_mean,
                                    'tf_CV2Median': CV2_median,
                                    'tf_CV': CV,
                                    'tf_Ir': IR,
                                    'tf_Lv': Lv,
                                    'tf_LvR': LvR,
                                    'tf_LcV': LcV,
                                    'tf_Si': SI,
                                    'tf_skw': skw,
                                    'wf_MaxAmpNorm': None,
                                    'wf_MaxAmp': None,
                                    'wf_MaxAmpTime': None,
                                    'wf_MinAmpNorm': None,
                                    'wf_MinAmp': None,
                                    'wf_MinAmpTime': None,
                                    'wf_Duration': None,
                                    'wf_PosHwDuration': None,
                                    'wf_NegHwDuration': None,
                                    'wf_Onset': None,
                                    'wf_OnsetTime': None,
                                    'wf_End': None,
                                    'wf_EndTime': None,
                                    'wf_Crossing': None,
                                    'wf_CrossingTime': None,
                                    'wf_Pk10': None,
                                    'wf_Pk10Time': None,
                                    'wf_Pk90': None,
                                    'wf_Pk90Time': None,
                                    'wf_Pk50': None,
                                    'wf_Pk50Time1': None,
                                    'wf_Pk50Time2': None,
                                    'wf_Tr10': None,
                                    'wf_Tr10Time': None,
                                    'wf_Tr90': None,
                                    'wf_Tr90Time': None,
                                    'wf_Tr50': None,
                                    'wf_Tr50Time1': None,
                                    'wf_Tr50Time2': None,
                                    'wf_PkTrRatio': None,
                                    'wf_DepolarizationSlope': None,
                                    'wf_RepolarizationSlope': None,
                                    'wf_RecoverySlope': None,
                                    'wf_RiseTime': None,
                                    'wf_PosDecayTime': None,
                                    'wf_FallTime': None,
                                    'wf_NegDecayTime': None,
                                    'wf_EndSlopeTau': None}, index=[0])

        df_qm_temp = pd.merge(unit_quality_metrics, unit_temporal_features, left_on='Unit', right_on='Unit')
        df_qm_temp.to_csv(f'{paths[1]}/Sample-{sample}-unit-{unit}.csv', index=False)
        print(f'{paths[1]}/Sample-{sample}-unit-{unit}.csv file successfully saved!')

