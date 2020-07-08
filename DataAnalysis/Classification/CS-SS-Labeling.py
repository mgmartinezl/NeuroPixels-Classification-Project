from pathlib import Path
import re
import pandas as pd
import ast
from rtn.npix.gl import get_units, load_units_qualities
from rtn.npix.corr import ccg, StarkAbeles2009_ccg_significance, ccg_sig_stack, gen_sfc, KopelowitzCohen2014_ccg_significance
from rtn.npix.plot import plot_ccg
import matplotlib.pyplot as plt

data_sets = [

    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1'  # DONE
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1' # DONE
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1' # FP PRESENT
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1' # DONE
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1' # TSV CLUSTER FILE!
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1' # DONE
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1' # DONE
    'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1' # FP PRESENT
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1' # FP PRESENT
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1' # FP PRESENT
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1' # DIDNT WORK
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1' # DONE
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1' # DONE
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1' # DONE
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1' # DONE
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1' # DONE
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1' # DONE
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2' # DONE
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1' # DONE
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1' # FP PRESENT
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1' # DONE
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1' # DONE
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1' # DONE
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1' # DONE
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1' # DONE
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1' # DONE
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1' # DONE
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1' # DONE
    # 'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1' # DONE
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1' # DONE
    # 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1' # DONE
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1' # DONE
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1' # DONE
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1' # DONE
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1' # DONE
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2' # DONE
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe2' # DIDNT WORK
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1' # DONE
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1' # DONE
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1' # DONE
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2' # DONE
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1' # DONE
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2' # DONE
    # 'D:/Recordings/20-15-06_DK186/20-15-06_DK186_probe1' # DONE

]

chan_range = [170, 384]
save_path = 'CS-SS'

# Params
cbin = 0.5
cwin = 100
pause_dur = 5
again = False
# test = 'Poisson_Stark'
test = 'Normal_Kopelowitz'

for dp in data_sets:

    cl_grp = load_units_qualities(dp)

    sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", dp)[0]

    # %% Find CCGs with long pause (at least 5ms)
    # ctx_units = get_units(dp, quality='good', chan_range=chan_range).tolist()

    # Extract good units of current sample
    all_good_units = pd.read_csv('C:/Users/NeuroPixels/PycharmProjects/NeuroPixels-Classification/DataPreprocessing/Images/Pipeline/GoodUnitsAllRecordings.csv')
    ctx_units = ast.literal_eval(all_good_units[all_good_units['Sample'].str.match(sample_probe)]['Units'].values[0])
    n_consec_bins = int(pause_dur // cbin)

    # Use the same 'name' keyword to ensure that ccg stack is saved to routines memory
    ccg_sig_05_100, ccg_sig_u, sfc = ccg_sig_stack(dp, ctx_units, ctx_units, cbin=cbin, cwin=cwin, name='ctx-ctx',
                                                   p_th=0.02, n_consec_bins=n_consec_bins, sgn=-1, fract_baseline=4. / 5,
                                                   W_sd=10, test=test, again=again, ret_features=True)

    sfc1 = gen_sfc(dp, corr_type='cs_pause', metric='amp_z', cbin=cbin, cwin=cwin, p_th=0.02, n_consec_bins=n_consec_bins,
                   fract_baseline=4. / 5, W_sd=10, test=test,
                   again=again, againCCG=again, units=ctx_units, name='ctx-ctx')[0]

    df = pd.DataFrame(columns=['unit', 'ss', 'cs'])

    for i in sfc1.index:
        cs, ss = sfc1.loc[i, 'uSrc':'uTrg'] if sfc1.loc[i, 't_ms'] > 0 else sfc1.loc[i, 'uSrc':'uTrg'][::-1]
        plot_ccg(dp, [cs, ss], 0.5, 100, normalize='Hertz')
        c = ccg(dp, [cs, ss], 0.5, 100, normalize='Counts')[0, 1, :]
        # StarkAbeles2009_ccg_significance(c, 0.5, 0.02, 10, -1, 10, True, True, True)

        plt.figure()
        KopelowitzCohen2014_ccg_significance(c, cbin=0.5, cwin=100, p_th=0.02, n_consec_bins=n_consec_bins, sgn=-1,
                                             fract_baseline=4. / 5, law='Normal', multi_comp=False,
                                             bin_wise=False, ret_values=True,
                                             only_max=True, plot=True)

        df = df.append({'unit': cs, 'ss': 0, 'cs': 1}, ignore_index=True)
        df = df.append({'unit': ss, 'ss': 1, 'cs': 0}, ignore_index=True)


    df.drop_duplicates(inplace=True)
    df.sort_values(by=['unit'], inplace=True)
    df = df.astype(int)

    df.to_csv(Path(save_path, f'SS-CS-labels-{sample_probe}.csv'))

