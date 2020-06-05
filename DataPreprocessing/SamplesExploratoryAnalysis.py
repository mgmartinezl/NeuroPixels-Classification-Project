from DataPreprocessing.AuxFunctions import plot_units_composition
import os.path as op;opj=op.join

PkC = [
    'F:/data/PkC/18-02-15_DK107/18-02-15_DK107_probe1',  # PROBE ERROR
    'F:/data/PkC/18-04-03_DK119/18-04-03_DK119_probe1',  # META BAD
    'F:/data/PkC/18-04-06_DK120/18-04-06_DK120_probe1',  # META BAD
    'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',  # GOOD
    'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',  # GOOD
    'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',  # META BAD
    'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',  # GOOD
    'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1'  # GOOD
]

GrC = [
    'F:/data/GrC/18-02-21_DK113/18-02-21_DK113_probe1',  # META BAD
    'F:/data/GrC/18-04-03_DK117/18-04-03_DK117_probe1',  # META BAD
    'F:/data/GrC/18-07-11_DK118/18-07-11_DK118_probe1',  # META BAD
    'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',  # GOOD
    'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',  # GOOD
    'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',  # META BAD
    'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',  # GOOD
    'F:/data/GrC/18-12-18_YC009/18-12-18_YC009_probe1',  # META BAD
    'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',  # GOOD
    'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',  # GOOD
    'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',  # META BAD
    'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',  # GOOD
    'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',  # META BAD
    'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',  # GOOD
    'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1'  # META BAD
]

GoC = [
    'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe2'  # GOOD but weird
]

MFb = [
    'F:/data/MFB/18-02-16_DK110/18-02-16_DK110_probe1',
    'F:/data/MFB/18-02-16_DK111/18-02-16_DK111_probe1',
    'F:/data/MFB/18-02-26_DK112/18-02-26_DK112_probe1',
    'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',  # GOOD
    'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',  # META BAD
    'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',  # GOOD
    'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',  # GOOD
    'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',  # GOOD
    'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',  # META BAD
    'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',  # GOOD
    'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',  # GOOD
    'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',  # GOOD
    'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1'  # GOOD
]

MLI = [
    'F:/data/MLI/18-02-21_DK114/18-02-21_DK114_probe1',
    'F:/data/MLI/18-02-28_DK115/18-02-28_DK115_probe1',
    'F:/data/MLI/18-07-12_DK122/18-07-12_DK122_probe1',
    'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1',
    'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',
    'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1'  # META BAD
]

PkC_gup = plot_units_composition(PkC)
PkC_gup.figure.savefig("PkC-units-prop-by-quality.png")

GrC_gup = plot_units_composition(GrC)
GrC_gup.figure.savefig("GrC-units-prop-by-quality.png")

MFb_gup = plot_units_composition(MFb)
MFb_gup.figure.savefig("MFb-units-prop-by-quality.png")

MLI_gup = plot_units_composition(MLI)
MLI_gup.figure.savefig("MLI-units-prop-by-quality.png")

