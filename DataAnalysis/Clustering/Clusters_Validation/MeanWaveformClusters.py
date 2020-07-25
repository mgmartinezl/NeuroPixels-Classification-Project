"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: QualityCheckPipeline.py
Description: implementation of quality metrics on units and chunks to choose the right ones for the next stage in the
classification pipeline.

"""

from DataPreprocessing.AuxFunctions import *
import pandas as pd
from rtn.npix.gl import get_units
from rtn.npix.spk_t import trn
from rtn.npix.corr import acg
from rtn.npix.spk_wvf import wvf, get_peak_chan
import matplotlib.pyplot as plt
import os
import re
import ast
import numpy as np

np.seterr(all='ignore')

data_sets = [

    # Yellow cluster
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1'

    # Blue cluster
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1'  # MLI

    # Red cluster
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # #'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',   ## REVIEW
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # #'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',   ## REVIEW
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # #'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',    ## REVIEW
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1',
    # 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1'

    # Gray cluster
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1'

    # NEW CLUSTERS
    # Purkinje
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',

    # CS
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1'

    # Gr
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',
    #
    # # Go+MF
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',

    # # MLI
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1',

    # Unknown
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',

    # GoMF subcluster 1
    'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',

    # GoMF subcluster 2
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',

    # GoMF subcluster 3
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',

    # GoMF subcluster 4
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1'

    # MLI subcluster 1
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1',

    # MLI subcluster 2
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1',

    # MLI subcluster 3
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1',

    # MLI subcluster 4
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',

    # MLI subcluster 5
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',

    # MLI subcluster 6
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1',

    # Pk subcluster 1
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',

    # Pk subcluster 2
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',

    # Pk subcluster 3
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',

    # # Pk subcluster 4
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',

    # CS subcluster 1
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',

    # CS subcluster 2
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # #'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',

    # Gr subcluster 1
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',

    # Gr subcluster 2
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1',
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1',
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1',
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1',
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2',
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1',

    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1',
    # 'D:/Recordings/20-15-06_DK186/20-15-06_DK186_probe1',
    # 'D:/Recordings/20-27-06_DK187/20-27-06_DK187_probe1'

    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1'

]

units = [

    # Yellow cluster
    # [125, 155, 159, 190, 194, 200, 205, 206, 209, 258, 292, 302, 46, 70, 76, 78, 99],
    # [12, 16, 35, 36, 72, 86, 94, 95, 103, 106, 123, 129, 140, 147, 157, 163, 169, 330],
    # [58, 81, 132, 160, 169, 173, 190, 196, 208, 209, 218, 222, 223, 229, 332, 393],
    # [22, 69, 73, 107],
    # [107, 133, 150, 210],
    # [8, 31, 98, 159, 225, 388],
    # [30, 49, 54, 81, 95, 149, 152, 170, 196, 240, 248, 272, 329, 344],
    # [13, 14, 17, 25, 29, 39, 65, 75, 88, 103, 105, 113, 124, 127, 129, 180, 215, 234],
    # [86, 113, 193, 297, 402],
    # [76, 77, 163],
    # [39, 49, 53, 55, 67, 78, 85, 87],
    # [371],
    # [44, 336],
    # [554],
    # [38, 51, 361, 378, 498],
    # [0, 19, 49, 95, 134, 135, 136, 166, 171, 173, 178, 214, 292, 334, 404, 427, 490, 511, 526, 540, 553, 586, 846, 871, 881, 884, 927],
    # [72, 90],
    # [40, 41, 75, 87, 98, 104, 221, 267, 273, 281, 283, 285, 286, 389],
    # [2],
    # [147, 211, 376, 391, 409, 417, 418, 441, 481, 842],
    # [103, 153, 156, 270, 275, 481, 504, 602, 692],
    # [69, 91, 138, 145, 206, 213, 220, 222, 224, 243, 245, 261, 280, 438, 446, 454, 456, 458, 462, 493],
    # [7, 22],
    # [16],
    # [132, 190, 194, 201, 208],
    # [171, 172],
    # [12, 79, 162, 163, 175, 179, 182, 427, 434, 449, 464, 466, 571, 621, 687, 736],
    # [76, 240, 287],
    # [9, 32, 38, 74, 78, 106, 109, 123, 145, 185, 203, 209, 212, 213, 250, 255, 258, 259, 287, 289],
    # [31, 35, 42, 56, 75, 175, 176, 184, 185, 188, 202],
    # [24, 31, 33, 37, 210, 218, 233, 402],
    # [10, 66, 94, 119, 148, 185, 205, 219, 267, 317, 370, 411, 424, 434, 568, 577],
    # [8, 39, 59, 66, 73, 74, 106, 108, 129, 146, 257, 414, 436, 439, 473],
    # [17, 23, 27, 36, 46, 155, 157, 280],
    # [84, 237, 240, 294]

    # Blue cluster
    # [134, 195, 204, 396],
    # [10, 180],
    # [162, 247],
    # [65, 164],
    # [0, 50],
    # [184, 376],
    # [94, 110, 126, 287],
    # [1, 22],
    # [30, 46],
    # [680, 874, 912, 1799],
    # [18, 210],
    # [328],
    # [160, 171,421,430],
    # [103,173,308,874],
    # [4,200,202,208,211],
    # [65,327,417],
    # [13, 52, 61, 165],
    # [102],
    # [17, 25,184,233,249,279,358,564],
    # [42, 535],
    # [107],
    # [26]

    # Red cluster
    # [188,198,201,221,291,398,48],
    # [144,149,153,158,170,255],
    # [0,170,152,181,189,230,239,300,377,54,641,92],
    # [74],
    # [130,35,43],
    # [42,134,247,244],
    # [26,111,139,280,286,364,95],
    # [0,2,34],
    # [651],
    # [351,847,850,851,864,882,281,452,642,643],
    # [162,210,211,99,1020,1058,276,296,299,470,497,529,772,904,906,944,972,167,216,429,825,834,849,876],
    # [104,322,532,867,923],
    # [136,146,451,486,5,513,598],
    # #[155,193,284,433,481,577,602,698,740,741,754,759,857,862,89,918,939],  # r
    # [286,163,55],
    # #[13,202,377,104,83,351,452,433,81,115,273,295,237,197,215,226,227,257,339,345,348,349,352,355,434,435,45,454,471,70,96,97],  # r
    # [30,41,34,186,190,228,232,29,36],
    # [103,429,389,237,28],
    # #[102,417,276,343,382,108,115,129,273,294,295,326,342,390,439,440,441,442,635,68],  # r
    # [103,417,48,111,43,21,101,105,137,147,309,32,33,330,372,382,389,394,406,411,415,416,427,428,432,437,49,541,574,579,81],
    # [249,7,228,196,234,250,253,6,62],
    # [22,19,21],
    # [104,19,20,373,41,66,82,83,86],
    # [272,314,99],
    # [154],
    # [246,824,928,472,438,400],
    # [168,304,547,558,563,898],
    # [41,86,95,374,485,80],
    # [396,329,43,79],
    # [27],
    # [233,249,279,296,242,343,448,459,464],
    # [122],
    # [152,122,114,78],
    # [27],
    # [7]

    # Gray cluster
    # [227,267],
    # [37],
    # [295,186,184,194,298],
    # [26],
    # [37],
    # [243,278],
    # [22,352,311,487,501,570,590,617,627],
    # [310],
    # [59],
    # [142],
    # [26,598,442,599,69],
    # [119, 654],
    # [104,702,732,91],
    # [624],
    # [165],
    # [68,228,301,305],
    # [329,330,306,323,324,353,354,356],
    # [41,241,546],
    # [30,188,230,239,130,272,246,242,273,237,232,194,69,205,151,133,150,169,182,183,203,208,212,218,219,220,231,248,251],
    # [47],
    # [62, 119],
    # [527],
    # [205],
    # [241, 492],
    # [129],
    # [158,145,151]

    # NEW CLUSTERS
    # Purkinje
    # [54,83,166,177,263,266,272,281,296,298,303,304,315,357,358,359,362,376,377,565],
    # [31,111,142,205,210,327],
    # [57,102,201,231,257,292,311,485],
    # [60,83,128,130,200],
    # [52,177],
    # [11,12,15,24,212,268,303,313,317,320,338,346,371,375,386],
    # [15,125,297,298,301,420,437,439,441,453,455,473,519,524,564,569,619,625,633,635],
    # [104,117,121,133,140,144,151,152,153,198,276,277,319,394,426],
    # [49,69,74,78,94,102,158,167,169,241,243,244,245,253,257,330,356,364],
    # [20,26,27,31,33,35,43,73,95,121,160,192,234],
    # [17,30,60,214,270],
    # [17,34,57,62,64,575],
    # [35,48,88,94,97,354,392,393,401,590,606,608,610,857,881,929,967,1026,1030,1081],
    # [67,71,79,84,100,119,128,140,141,147,176,204,290,436,631,794],
    # [59,65,67,83,100,102,142,161,211,268,274,298,301,314,527,605,714,895,953],
    # [17,20,22,25,47,50,56,65,66,68,74,83,91,101,119,126,131,132,154,218,224,227,239,243,377,378,418,437,443,490,501,509,539,544,549,558,666,716,812,814,815,848,910,938,959,964,979,993],
    # [9,59,65,66,85,92],
    # [36,39,46,49,61,147,150,187,188,201,203,208,209,325,342,344,346,350,462],
    # [6,18,19],
    # [2,3,16,17,19,21,35,37,74,111,124,127,163,177,235,243,247,259,268,278,302,381,393,529,596,600,605,652,666,670,745,773,844,889,950],
    # [53,85,94,226,232,241,242,249,251,252,254,256,281,725,728,729,730,735],
    # [49,58,60,62,64,74,83,151,156,161,175,186,192,204,238],
    # [3,6,8,19,23,26,27,28,29,30,32,37,47,86],
    # [19,23,85,88,97],
    # [2,3,14,28,33,39,41,68,69,71,72,73,81,86,89,97,106,121,131,136,142,148,162,172,175,176,180,183,304,329,330,331],
    # [3,6,7,11,12,13,16,19,20,65,71,75,81,94,104,121,144,169,157,180,206,278,280],
    # [4,19,25,29,37,57,61,66,84,101,140,200,210,219,227,261,270,290,276,319,368,373,398,403,444,450,451,553,718,796,802,819,831,833,839,841,842,971],
    # [48,53,59,67,71,140,141,142,253,277,278,280,284],
    # [8,29,51,162,178,280],
    # [7,9,10,15,32,40,46,85,89,98,115,123,128,133,142,144,150,157,154,167,191,192],
    # [60,93,213,361],
    # [15,18,19,20,21,23,28,31,36,48,81,92,121,175,191,218,314,376,346,445,549,622],
    # [197,209,217,220,263,470,478],
    # [47,51,81,113,120,132,134,156,162,173,192,224,241],

    # Purkinje half-data
    # [166, 177, 263, 266, 272, 281, 296, 298, 303, 304, 315, 357, 358, 359, 362, 376, 377, 54, 565, 83],
    # [111, 142, 205, 210, 31, 327],
    # [102, 201, 231, 257, 292, 311, 485, 57],
    # [60, 83, 128, 130, 200],
    # [52, 177],
    # [11,12,15,24,212,268,303,313,317,320,338,346,371,375,386],
    # [15,125,297,298,301,420,437,439,441,453,455,473,519,524,564,569,619,625,633,635],
    # [104,117,121,133,140,144,151,152,153,198,276,277,319,394,426],
    # [49,69,74,78,94,102,158,167,169,241,243,244,245,253,257,330,356,364],
    # [20,26,27,31,33,35,43,73,95,121,160,192,234],
    # [104,117,121,133,140,144,151,152,153,198,276,277,319,394,426],
    # [49,69,74,78,94,102,158,167,169,241,243,244,245,253,257,330,356,364],

    # # CS
    # [20,26,27,44,69,71,82,86,89,93,129,130,133,135,137,139,149,154,156,158,162,163,193,230,233,234,235,238,239,240,242,245,254,256,261,465,494],
    # [59,21,39,41,51,60,272,275],
    # [122,124,127,133,134,135,137,139,187,224,232,246,277,343,355,381,457,560],
    # [28,63,90,92,102],
    # [84,93,96,115,117,120,121,123,126,169,36,67,68,70,71,73,77,78,79,80,81],
    # [258],
    # [366,445,447,25,7,5],
    # [85,86,102,61,48,45,131,354],
    # [83,81,69,93],
    # [654,14,6,684,797,814,1324],
    # [536,427,425],
    # [905,887,865],
    # [85,9,3,104,127,977,1047],
    # [305,186,141,539],
    # [22,20,6,400,472],
    # [203,949],
    # [241],
    # [86,64,63],
    # [126],
    # [588,586,30],
    # [25],

    # # Gr
    # [24,134,195,204,396],
    # [10],
    # [162,247],
    # [65,164],
    # [0,50],
    # [184],
    # [94,110,126,287],
    # [1,22],
    # [30,46,370],
    # [912],
    # [680],
    # [18],
    # [328],
    # [160,421,430],
    # [103,173,308,874],
    # [4,200,202,208,211],
    # [65,223,327,417],
    # [13,52,165],
    # [17,102,165],
    # [17,25,184,233,249,279,315,358,564],
    # [42, 535],
    # [107],
    # [26],
    #
    # # Go+MF
    # [188,198,201,221,398],
    # [149,153,170,255],
    # [0,54,92,152,170,239,295,300],
    # [74],
    # [35,43],
    # [19,20,41,66,82,83,86,104,373],
    # [42,134,244],
    # [26,95,111,280,286,364],
    # [99,272,314],
    # [0,2,34],
    # [154],
    # [246,400,438,472,824,928],
    # [168,304,547,558,898],
    # [651],
    # [351,847,850,851,864,882],
    # [452,642,643],
    # [99,162,276,299,529,944,972,1020,1058],
    # [167,210,211,216,429,825,834,849,876],
    # [41,80,86,95,374,485],
    # [43,79,329,396],
    # [27],
    # [104,322,532,867],
    # [5,136,146,451,486,513,598],
    # [233,242,249,296,448,459,464,492],
    # [78,114,122,152],
    # [89,155,193,284,433,481,577,602,740,741,754,759,857,862,918,939],
    # [55,163,286],
    # [13,352,355,433,435,452,454,471,45,70,81,83,96,97,115,197,202,226,227,257,273,295,339,345,348,349,351],
    # [29,30,36,41,186,190,228,232],
    # [28,103,237,389,429],
    # [68,102,108,115,129,273,276,294,295,326,342,382,390,440,417,441,442],
    # [21,32,33,43,48,49,81,103,105,111,137,147,309,372,330,382,389,394,406,411,415,416,417,427,428,541,574,579],
    # [6,7,62,196,220,228,231,234,249,250,253],
    # [27],
    # [19,21,22],

    # # MLI
    # [45,46,48,70,76,78,99,115,125,155,159,189,190,194,200,205,206,209,241,258,288,291,292,302,369,399],
    # [12,16,22,26,35,36,72,86,94,95,103,106,123,129,140,144,147,157,158,163,169,173,180,330],
    # [58,78,80,81,97,101,132,160,169,173,175,181,189,190,196,208,209,218,222,223,229,230,332,377,393,641],
    # [22,69,73,107],
    # [107,130,133,150,210],
    # [8,31,98,159,225,388],
    # [28,30,49,54,81,95,120,149,152,170,196,240,247,248,272,329,344,376],
    # [13,14,17,25,29,39,65,75,88,103,105,113,124,127,129,139,180,215,234,240,272,281,283,288,290,329,336,373,566,570,594,611,617],
    # [2,50,86,97,102,113,191,193,297,402],
    # [76,77,79,163,174],
    # [36,39,49,53,55,67,78,85,87],
    # [371],
    # [21,44,336],
    # [554,563],
    # [7,27,38,51,58,71,361,378,498],
    # [23,27,31,34,53,60,98,133,313,363,381,846,848,867,921,934,975,1799],
    # [56,63,68,72,73,129,172,190,237,281,283,331,439,440,521,641,652,658,663,741,774,874,931],
    # [19,49,95,134,136,210,214,292,296,470,490,493,497,511,526,540,772,871,904,906,927],
    # [0,26,135,136,166,171,173,178,334,404,417,427,511,553,586,846,881,884],
    # [72,90,530],
    # [40,41,75,87,98,104,184,221,253,258,267,273,281,283,285,286,323,389],
    # [2],
    # [147,211,370,376,391,409,417,418,441,481,842,923],
    # [103,153,156,171,270,275,481,504,515,602,692],
    # [69,91,138,145,206,213,220,222,224,241,243,245,261,267,279,280,343,438,446,454,456,458,462,493],
    # [7,22],
    # [16],
    # [1,6,7,54,122,132,190,194,201,208],
    # [171,172],
    # [12,79,91,162,163,166,175,179,182,427,434,449,464,466,480,571,621,687,698,736,806],
    # [76,151,160,240,287],
    # [7],
    # [4,9,32,38,74,78,104,106,109,123,145,185,203,205,209,212,213,215,237,250,255,258,259,287,289,293,301,305,358,371,377,434,447],
    # [31,34,35,42,56,61,75,175,176,184,185,187,188,199,200,202],
    # [24,31,33,37,150,210,218,233,402],
    # [10,66,87,94,119,142,148,185,205,219,241,267,317,343,352,370,411,424,439,434,456,554,568,577,635],
    # [8,59,66,73,74,78,101,106,108,129,146,257,414,432,436,437,439,473],
    # [17,23,27,30,36,43,46,52,155,157,280],
    # [39,43],
    # [84,237,240,294,334,173],

    # Unknown
    # [227,267],
    # [37],
    # [184, 186, 194, 298],
    # [26],
    # [37],
    # [62,119],
    # [243,278],
    # [22,311,352,487,501,590],
    # [310],
    # [59],
    # [142],
    # [69,442,598,599],
    # [527],
    # [205],
    # [119,654],
    # [91,104,702,732],
    # [129],
    # [145,151,158],
    # [624],
    # [165],
    # [68,228],
    # [306,323,324,329,330,353,354,356],
    # [41,546],
    # [69,130,133,150,151,169,182,183,188,194,203,205,208,212,218,219,230,232,237,239,242,246,248,251,273,272],
    # [47]

    # GoMF subcluster 1
    [188, 221, 398],  # NO
    [153, 170],  # NO
    [54, 92],  # NO
    [74],  # NO
    [43],
    [86],  # NO
    [244],  # NO
    [26, 95, 111, 280],  # NO
    [272, 314],  # NO
    [34],  # NO
    [154],  # NO
    [246, 400, 824],  # NO
    [574],
    [351, 847, 850, 851],
    [452, 642],
    [529, 972],
    [167, 211, 216, 429, 825, 834, 849, 876],
    [41, 80, 485],
    [329, 396],
    [136, 451],
    [296],
    [433, 481, 754, 759, 857],
    [55, 286],
    [70, 83, 115, 197, 273, 348, 352, 355],
    [30, 36],
    [108, 294, 342, 382, 442],
    [372],
    [7],
    [19, 21]

    # GoMF subcluster 2
    # [198],
    # [255],
    # [170,239,295],
    # # [35],
    # [104],
    # [42],
    # [286, 364],
    # [472],
    # [558, 898],
    # [864],
    # [99,276,299,944,1058],
    # [86,95,374],
    # [79],
    # [104,322,532],
    # [5,146,513,598],
    # [233,249,448,464,492],
    # [152],
    # [155,193,284,740,862],
    # [202,257,339,349,351,435,454,471],
    # [41,190,228],
    # [103,237],
    # [102,129,273,276,295,326,417,440],
    # [32,33,49,147,382,406,427,428],
    # [6,196,220,228,231,249,250,253],
    # [27]

    # GoMF subcluster 3
    # [201],
    # [0,152,300],
    # [373],
    # [99],
    # [2],
    # [438,928],
    # [168],
    # [643],
    # [162,1020],
    # [27],
    # [867],
    # [486],
    # [242,459],
    # [78,114,122],
    # [89,577,602],
    # [13,226,295,433,452],
    # [29,186,232],
    # [28, 429],
    # [68,115,390],
    # [43,81,103,105,137,309,330,389,394,411,415,416,541,574,579],
    # [234]

    # GoMF subcluster 4
    # [149],
    # [19,20,41,66,82,83],
    # [134],
    # [0],
    # [304],
    # [651],
    # [882],
    # [210],
    # [43],
    # [741,918,939],
    # [163],
    # [45,81,96,97,227,345],
    # [389],
    # [441],
    # [21,48,111,417],
    # [62]

    # MLI subcluster 1
    # [45,46,70,76,78,99,	115,155,189,190,241,258,288,292,302,369,399],
    # [22,26,36,72,123,140],
    # [78,80,97,101,160,169,175,223,393],
    # [107],
    # [107,150,210],
    # [8,225],
    # [28,49,95,149,248],
    # [13,17,25,29,103,240,272,283,290],
    # [191],
    # [174],
    # [53,78,87],
    # [371],
    # [21],
    # [554],
    # [7,27,51,58,71,361],
    # [23,27,31,34,313,363,381,921],
    # [63,237,283,331,521],
    # [49,214,526,540],
    # [166,334,417,846],
    # [72, 530],
    # [75,87,104,253,258,267,283,323,389],
    # [418,441],
    # [515,602],
    # [222,261,267,280],
    # [22],
    # [54,201],
    # [91,162,163,166,179,182,480,806],
    # [4,203,205,250,293],
    # [75,185,188,200,202],
    # [150],
    # [87,142,219,267,568,577],
    # [74,78,146,257,436,473],
    # [27,36,43,52],
    # [39,43],
    # [84],

    # MLI subcluster 2
    # [125,194,206],
    # [12,86,103,129,163,173,180,330],
    # [132,173,208,209,229],
    # [22,69,73],
    # [133],
    # [30,54,120,376],
    # [65,75,129,234],
    # [50,97,297],
    # [79,163],
    # [36,39,49,67],
    # [336],
    # [38,498],
    # [1799],
    # [56,874],
    # [95,136,210,493],
    # [171,427,586,881],
    # [417],
    # [153,171],
    # [145,213,220,224,245,438,462],
    # [6,7,208],
    # [79,175,449,464,736],
    # [240,287],
    # [109,289],
    # [35,61,184,199],
    # [24,37,210],
    # [185,352,434,456],
    # [8,108,129,439],
    # [237,294],

    # MLI subcluster 3
    # [159,200,205,209],
    # [16,35,106],
    # [196,218,332],
    # [31,98,388],
    # [240,329,344],
    # [39,88,113,127,180,281,594,611],
    # [193],
    # [85],
    # [98,846,934],
    # [129],
    # [511, 927],
    # [404,553],
    # [147,211,376,481],
    # [103,270,275,481],
    # [138,446,493],
    # [7],
    # [12],
    # [76],
    # [145,212,259,358,371],
    # [56],
    # [10, 411],
    # [59,73],
    # [17,46],
    # [240],

    # MLI subcluster 4
    # [48,291],
    # [94,144,158,169],
    # [181,189,190,230,377,641],
    # [130],
    # [81,274],
    # [139],
    # [55],
    # [563],
    # [867],
    # [281,658,663,931],
    # [296,470,497,772,904,906],
    # [41,285],
    # [2],
    # [842,923],
    # [156],
    # [279,343,456],
    # [122],
    # [171,172],
    # [466,571,621,698],
    # [7],
    # [104,215,237,255,258,377,434],
    # [34,42,175,176],
    # [31,233],
    # [119,343,424,439,635],
    # [101,106,432,437],
    # [280]

    # MLI subcluster 5
    # [147],
    # [81],
    # [159],
    # [152,272],
    # [215,288,329,336,566,570,617],
    # [113,402],
    # [77],
    # [848],
    # [73,439,440,641,774],
    # [19,134,490],
    # [0,26,135,173,884],
    # [90],
    # [273,281,286],
    # [409],
    # [504],
    # [91,241,458],
    # [16],
    # [132, 194],
    # [427],
    # [32,209,301,305],
    # [31],
    # [33,218],
    # [66,148,241,317],
    # [66],
    # [30,155,157],

    # MLI subcluster 6
    # [95,157],
    # [58,222],
    # [170,196],
    # [14,105,124,373],
    # [2,86,102],
    # [76],
    # [44],
    # [378],
    # [53,60,133,975],
    # [68,72,172,190,652,741],
    # [292,871],
    # [136,178,511],
    # [40,98,184,221],
    # [370,391],
    # [692],
    # [69,206,243,454],
    # [1,190],
    # [434,687],
    # [151,160],
    # [9,38,74,78,106,123,185,213,287,447],
    # [187],
    # [402],
    # [94,205,370,554],
    # [414],
    # [23],
    # [334],

    # Pk subcluster 1
    # [166,296,376,377],
    # [111,205,210,327],
    # [57,102,201,231,257,485],
    # [130,200],
    # [212,268,317,346,371,386],
    # [297,298,473,519,524,564,619,625],
    # [117,121,133,140,144,151,152,276,277,319,394],
    # [102,158,257,330,364],
    # [20,26,27,31,43,73,95,121,160],
    # [17,30],
    # [575],
    # [608,1081],
    # [67,71,79,140,141,290,631],
    # [100,211,714],
    # [47,65,66,68,74,91,101,126,154,224,227,243,443,509,544,558,815,993],
    # [85],
    # [39,46,49,147,150,187,325],
    # [6,18,19],
    # [2,3,16,21,35,37,74,111,124,177,278,381,596,600,605,666,773,844,889,950],
    # [232,281],
    # [58,60,62,64,83,151,192,238],
    # [19,26,27,28,30,37],
    # [23],
    # [28,33,39,41,69,71,86,97,131,142,175,176,183,330,331],
    # [6,7,11,12,13,16,19,20,65,144,157,169,206,278,280],
    # [261,270,444,451,718,802,831],
    # [277,278,280,284],
    # [8,51,162],
    # [7,9,10,32,40,46,85,133,142],
    # [60,93],
    # [15,18,19,20,21,28,36,48,81,92,121,218,549],
    # [47,51,134,173,192,241]

    # Pk subcluster 2
    # [54,177,263,266,357,358,359,565],
    # [292],
    # [128],
    # [52,177],
    # [338],
    # [420,437,439,455,633,635],
    # [153,198,426],
    # [243,245,356],
    # [192,234],
    # [97,401,1026],
    # [67,83,102,142,953],
    # [119,218,239,490,666,910,959],
    # [65],
    # [127,163,243,247,259,268,652,670,745],
    # [85,94,252,725,728,729,730],
    # [156,161],
    # [3,8,86],
    # [68,72,73,106,121,148,162,180,329],
    # [71,81,104,180],
    # [4,19,25,29,37,57,61,66,84,140,210,227,276,290,368,373,403,796,819,839,842],
    # [67,71],
    # [89,98,115,123,144,157],
    # [361],
    # [175],
    # [470],
    # [81,113,120,132,156,162,224]

    # Pk subcluster 3
    # [83,272,281,298,303,304,315,362],
    # [31,142],
    # [311],
    # [60,83],
    # [11,12,15,24,303,313,320,375],
    # [301,441,453,569],
    # [49,69,74,94,169,241,244,253],
    # [33,35],
    # [60,214],
    # [17,34,57,62,64],
    # [48,88,94,354,392,393,590,606,610,857,881,929,967],
    # [84,100,119,128,147,176,204,436,794],
    # [59,65,161,268,274,298,301,527,605,895],
    # [17,20,22,25,50,56,83,131,132,377,378,418,437,501,539,549,716,812,814,848,938,964,979],
    # [9,59,66,92],
    # [36,61,188,201,203,208,209,344,346,350,462],
    # [17,19,235,302,393,529],
    # [226,241,242,249,251,254,256,735],
    # [49,74,175,186,204],
    # [6,23,29,32,47],
    # [19,85,88,97],
    # [2,3,81,89,136,172],
    # [3,75,94,121],
    # [101,200,219,319,398,553,833,841,971],
    # [48,53,59,140,141,142,253],
    # [29,178,280],
    # [15,128,150,154,167,191,192],
    # [213],
    # [23,31,191,346,376,445,622],
    # [197,209,217,220,263,478]

    # Pk subcluster 4
    # [15,125],
    # [104],
    # [78,167],
    # [270],
    # [35,1030],
    # [314],
    # [342],
    # [53],
    # [14,304],
    # [450],
    # [314]

    # CS subcluster 1
    # [20,26,27,69,71,82,86,89,93,129,130,133,135,137,139,149,154,156,158,162,163,193,230,233,234,235,238,239,240,
    # 242,245,254,256,261,465,494],
    # [39,41,51,59,60,272,275],
    # [122,124,127,133,134,135,137,139,187,232,246,277,381,560],
    # [90,92],
    # [36,67,68,70,71,73,77,79,80,81,84,93,96,115,117,120,121,123,126,169],
    # [5,7,25,366,445],
    # [45,48,61,85,86,131],
    # [69,81,83,93],
    # [6,14,654,684,797,814,1324],
    # [425,427,536],
    # [865,887,905],
    # [3,9,85,104,127,977,1047],
    # [141,186,305,539],
    # [6,20,22,400,472],
    # [203],
    # [126]

    # CS subcluster 2
    # [44],
    # [21],
    # [224,343,355,457],
    # [28,63,102],
    # # [78],
    # [258],
    # [447],
    # [102,354],
    # [949],
    # [241],
    # [63,64,86],
    # [30,586,588],
    # [25]

    # Gr subcluster 1
    # [24,134,195,204],
    # [162,247],
    # [65],
    # [184],
    # [110,287],
    # [30,46],
    # [912],
    # [680],
    # [18],
    # [160,421],
    # [103,308],
    # [4,200,202,208,211],
    # [65],
    # [165],
    # [17,25,564],

    # Gr subcluster 2
    # [396],
    # [10],
    # [164],
    # [0,50],
    # [94,126],
    # [1,22],
    # [370],
    # [328],
    # [430],
    # [173,874],
    # [223,327,417],
    # [13,52],
    # [17,102,165],
    # [184,233,249,279,315,358],
    # [42,535],
    # [107],
    # [26]

    # Responsive Golgi
    # [228, 280],
    # [571,425,535,540,825],
    # [385,404,479,493]

    # # Responsive MLI
    # [334],
    # [294],
    # [240],
    # [237]

]

# Parameters
fs = 30000
c_bin = 0.2
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
t_waveforms = 80
again = False

# Collect for later...
l_cluster_trns = []
l_cluster_wvfs = []
l_cluster_acgs = []
l_cluster_amps = []
l_cluster_wvfs_samples = []

fig, axs = plt.subplots(2, 2, figsize=(10, 7))

n = 0

for dp in data_sets:

    sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", dp)[0]
    sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", dp)[0]
    print('***************************')
    print(f'Sample {data_sets.index(dp) + 1} of {len(data_sets)}')
    # print('Sample:', sample_probe)
    # cell_type = dp[8:11]

    print(sample_probe)

    # Load kilosort aux files
    amplitudes_sample = np.load(f'{dp}//amplitudes.npy')  # shape N_tot_spikes x 1
    spike_times = np.load(f'{dp}//spike_times.npy')  # in samples
    spike_clusters = np.load(f'{dp}//spike_clusters.npy')

    # Extract chunks for good units
    units_chunks = pd.read_csv(f'../Images/Pipeline/Sample_{sample_probe}/Sample-{sample_probe}-ChunksUnits.csv')
    print(units_chunks.tail())
    # Units in cluster
    # good_units = [125,155,159,190,194,200,205,206,209,258,292,302,46,70,76,78,99]
    good_units = units[data_sets.index(dp)]

    for unit in good_units:

        try:

            print('Unit >>', unit)

            n += 1

            # try:
            # print(f'Unit {good_units.index(unit)+1} of {len(good_units)}')

            # Extract chunks
            chunks = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['GoodChunks'][
                                            units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[0]])))

            chunks_wvf = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['WvfChunks'][
                                                units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[
                                                    0]])))

            # chunks_wvf = [3, 4]

            # Store quality metrics
            unit_quality_metrics = pd.read_csv(
                f'../Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files/Sample-{sample}-unit-{unit}.csv')

            # Retrieve peak channel for this unit
            peak_channel = unit_quality_metrics['PeakChUnit'][0]

            # Unit amplitudes
            amplitudes_unit = amplitudes_sample[spike_clusters == unit]
            spike_times_unit = spike_times[spike_clusters == unit]
            unit_mask_20 = (spike_times_unit <= samples_fr)
            spike_times_unit_20 = spike_times_unit[unit_mask_20]
            amplitudes_unit_20 = amplitudes_unit[unit_mask_20]

            # Unit ACG
            # try:
            block_ACG = acg(dp, unit, c_bin, c_win, subset_selection=[(0, unit_size_s)])
            x_block = np.linspace(-c_win * 1. / 2, c_win * 1. / 2, block_ACG.shape[0])
            y_block = block_ACG.copy()
            y_lim1_unit = 0
            yl_unit = max(block_ACG)
            y_lim2_unit = int(yl_unit) + 5 - (yl_unit % 5)

            # Sore ACG info
            l_cluster_acgs.append(np.array(y_block))

            # except Exception:
            #    pass

            # Read chunks...
            for i in range(n_chunks):
                if i in chunks:

                    # Chunk length and times in seconds
                    chunk_start_time = i * chunk_size_s
                    chunk_end_time = (i + 1) * chunk_size_s

                    # Chunk spikes
                    trn_samples_chunk = trn(dp, unit=unit,
                                            subset_selection=[(chunk_start_time, chunk_end_time)],
                                            enforced_rp=0.5, again=again)

                    l_cluster_trns.append(trn_samples_chunk)
                    # print(trn_samples_chunk)

                    # Chunk mask for times
                    chunk_mask = (i * chunk_size_s * fs <= spike_times_unit_20) & \
                                 (spike_times_unit_20 < (i + 1) * chunk_size_s * fs)
                    chunk_mask = chunk_mask.reshape(len(spike_times_unit_20), )

                    # Chunk amplitudes
                    amplitudes_chunk = amplitudes_unit_20[chunk_mask]
                    amplitudes_chunk = np.asarray(amplitudes_chunk, dtype='float64')

                    l_cluster_amps.append(amplitudes_chunk)
                    # print(l_cluster_amps)

                else:
                    continue

            unit_waves = []
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

                    chunk_wvf = chunk_wvf / -min(chunk_wvf)
                    l_cluster_wvfs.append(chunk_wvf)
                    unit_waves.append(chunk_wvf)

            # l_cluster_wvfs.append(np.mean(unit_waves, axis=0))

                # for wave in l_cluster_wvfs:
                #     plt.plot(wave, color='lightgray')

        except Exception:
            continue

    # Waveform of the whole sample (all units)
    # print(l_cluster_wvfs)
    # waveform_sample = np.mean(l_cluster_wvfs, axis=0)
    waveform_sample = np.mean(l_cluster_wvfs, axis=0)
    l_cluster_wvfs_samples.append(waveform_sample)

    # Plot every sample mean waveform
    axs[0, 0].plot(waveform_sample, color='lightgray')

    # print('AVERAGE >> ', np.nanmean(l_cluster_wvfs_samples, axis=0))

# plt.plot(np.mean(l_cluster_wvfs_samples, axis=0), color='salmon')
# plt.show()

# import sys
# sys.exit()
# *********************************************************************************************************

# Concatenate same cluster units' spikes together
trn_samples_cluster = np.concatenate(l_cluster_trns).ravel()

# Compute Inter-Spike Interval for the cluster
isi_cluster_clipped = compute_isi(trn_samples_cluster, quantile=exclusion_quantile)
isi_cluster_no_clipped = compute_isi(trn_samples_cluster)

# Estimate number of optimal bins for ISI
isi_bins = estimate_bins(isi_cluster_clipped, rule='Sqrt')
isi_log_bins = None

try:
    isi_log_bins = np.geomspace(isi_cluster_clipped.min(), isi_cluster_clipped.max(), isi_bins)
except ValueError:
    pass

# Compute temporal features for the cluster
mfr = mean_firing_rate(isi_cluster_clipped)
mifr, med_isi, mode_isi, prct5ISI, entropy, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, skw \
    = compute_isi_features(isi_cluster_clipped)

# Put waveforms together
# waveform_cluster = np.mean(l_cluster_wvfs, axis=0)

# Normalize mean waveform
# waveform_block = waveform_cluster / -min(waveform_cluster)

# Concatenate same cluster units' acgs together
acg_cluster = np.mean(l_cluster_acgs, axis=0)
x_cluster = np.linspace(-c_win * 1. / 2, c_win * 1. / 2, acg_cluster.shape[0])
y_cluster = acg_cluster.copy()
y_lim1_cluster = 0
yl_cluster = max(acg_cluster)
y_lim2_cluster = int(yl_cluster) + 6 - (yl_cluster % 5)

# Concatenate same cluster units' amps together
amps_cluster = np.concatenate(l_cluster_amps).ravel()
amps_cluster_clipped = remove_outliers(amps_cluster, exclusion_quantile)

# Mean Amplitude Cluster
ma_cluster = mean_amplitude(amps_cluster_clipped)
amps_cluster_bins = estimate_bins(amps_cluster, rule='Fd')

# Estimate optimal number of bins for Gaussian fit to amplitudes
# cluster_bins = estimate_bins(amps_cluster, rule='Fd')

# % of missing spikes per cluster
# x_c, p0_c, min_amp_c, n_fit_c, n_fit_no_cut_c, chunk_spikes_missing = gaussian_amp_est(amps_cluster, cluster_bins)

# *************************************************************************************
# Graphics!!

fig.suptitle(f'Cluster: {n} units', fontsize=9, color='#939799')

# [0, 0]
x1 = list(np.arange(len(waveform_sample), step=10))
x2 = [round(i / 30, 2) for i in x1]  # Samples to ms
plt.sca(axs[0, 0])
axs[0, 0].plot(np.nanmean(l_cluster_wvfs_samples, axis=0), color='dimgray')
plt.xticks(x1, x2)
axs[0, 0].set_ylabel("mV", size=9, color='#939799')
axs[0, 0].set_xlabel("ms", size=9, color='#939799')
axs[0, 0].set_title(f' \n \n \n Mean cluster waveform  \n', fontsize=9, loc='center', color='#939799')

# [0, 1]
axs[0, 1].hist(amps_cluster,
               bins=amps_cluster_bins,
               orientation='vertical',
               color='lightgray')

leg_dtr_0_1 = [f'Mean Amplitude: {ma_cluster}']
leg_0_1 = axs[0, 1].legend(leg_dtr_0_1, loc='best', frameon=False, fontsize=9)

for text in leg_0_1.get_texts():
    text.set_color("gray")

for lh in leg_0_1.legendHandles:
    lh.set_alpha(0)

axs[0, 1].set_title(f' \n \n \n Cluster amplitudes', fontsize=9, loc='center', color='#939799')

axs[1, 0].hist(isi_cluster_clipped / (1. / 1000), bins=100, color='lightgray')
axs[1, 0].set_xscale('log')
# axs[1, 0].set_xlim(left=0.1)
axs[1, 0].set_xlabel('Inter Spike Interval (ms) [log scale]')
leg_line_mfr = [f'MFR = {mfr} Hz \n']
leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)

for text in leg_1_0.get_texts():
    text.set_color("gray")

for lh in leg_1_0.legendHandles:
    lh.set_alpha(0)

axs[1, 1].bar(x=x_cluster, height=y_cluster, width=0.2, color='salmon', bottom=y_lim1_cluster)
axs[1, 1].set_ylim([y_lim1_cluster, y_lim2_cluster])
axs[1, 1].set_ylabel("Autocorrelation (Hz)", size=9, color='#939799')
axs[1, 1].set_xlabel("Time (ms)", size=9)
axs[1, 1].set_title(f' \n \n Autocorrelogram', fontsize=9, fontname="DejaVu Sans", loc='center',
                    color='#939799')

# FORMATTING
format_plot(axs[0, 0])
format_plot(axs[0, 1])
format_plot(axs[1, 0])
format_plot(axs[1, 1])
fig.tight_layout()
plt.show()
