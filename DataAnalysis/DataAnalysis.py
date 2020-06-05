import pandas as pd

keys = ['Pk-18-08-30_YC001_probe1-315',
        'Pk-18-08-30_YC001_probe1-303',
        'Pk-18-08-30_YC001_probe1-376',
        'Pk-18-08-30_YC001_probe1-227',
        'Pk-18-08-30_YC002_probe1-139',
        'Pk-18-08-30_YC002_probe1-86',
        'Pk-18-12-17_YC010_probe1-319',
        'Pk-18-12-17_YC010_probe1-314',
        'Pk-18-12-17_YC010_probe1-278',
        'Pk-18-12-17_YC010_probe1-276',
        'Pk-18-12-17_YC010_probe1-426',
        'Pk-18-12-17_YC010_probe1-259',
        'Pk-18-12-17_YC010_probe1-198',
        'Pk-18-12-17_YC010_probe1-394',
        'Pk-18-12-18_YC010_probe1-214',
        'Pk-18-12-17_YC010_probe1-356',
        'Pk-18-12-17_YC010_probe1-364',
        'Pk-18-12-17_YC010_probe1-241',
        'Pk-19-01-16_YC011_probe1-160',
        'Pk-19-01-16_YC011_probe1-192',
        'Pk-19-01-16_YC011_probe1-43',
        'MLI-19-12-13_YC007_probe1-674',
        'MLI-19-12-13_YC007_probe1-654',
        'MLI-19-12-13_YC007_probe1-334',
        'MLI-19-12-13_YC007_probe1-294',
        'MLI-19-12-13_YC007_probe1-240',
        'MLI-19-12-13_YC007_probe1-237',
        'MLI-19-12-13_YC007_probe1-674',
        'GrC-18-08-31_YC003_probe1-246',
        'GrC-18-08-31_YC003_probe1-247',
        'GrC-18-08-31_YC003_probe1-342',
        'GrC-19-08-14_YC015_probe2-722',
        'GrC-19-08-14_YC015_probe2-680',
        'GrC-19-08-14_YC015_probe2-850',
        'GrC-19-08-14_YC016_probe2-1013',
        'GrC-19-08-14_YC016_probe2-640',
        'MF-18-09-03_YC005_probe1-43',
        'MF-19-01-23_YC012_probe1-472',
        'MF-19-01-23_YC012_probe1-438',
        'MF-19-01-23_YC012_probe1-400',
        'MF-19-01-23_YC012_probe1-804',
        'MF-19-01-23_YC012_probe1-371',
        'MF-19-01-23_YC012_probe1-928',
        'MF-19-01-24_YC013_probe1-378',
        'MF-19-01-24_YC013_probe1-370',
        'MF-19-01-24_YC013_probe1-361',
        'MF-19-01-24_YC013_probe1-498',
        'Go-19-11-11_YC040_probe1-228',
        'Go-19-11-11_YC040_probe1-260',
        'Go-19-11-11_YC040_probe1-240',
        'Go-19-11-11_YC040_probe1-275',
        'Go-19-11-11_YC040_probe1-280'
]

Pk_Responsive = [
'18-08-30_YC001-315',
'18-08-30_YC001-303',
'18-08-30_YC001-376',
'18-08-30_YC001-227',
'18-08-30_YC002-139',
'18-08-30_YC002-86',
'18-12-17_YC010-319',
'18-12-17_YC010-314',
'18-12-17_YC010-278',
'18-12-17_YC010-276',
'18-12-17_YC010-426',
'18-12-17_YC010-259',
'18-12-17_YC010-198',
'18-12-17_YC010-394',
'18-12-18_YC010-214',
'18-12-17_YC010-356',
'18-12-17_YC010-364',
'18-12-17_YC010-241',
'19-01-16_YC011-160',
'19-01-16_YC011-192',
'19-01-16_YC011-43'
]

MLI_Responsive = [
        '19-12-13_YC007-674',
        '19-12-13_YC007-654',
        '19-12-13_YC007-334',
        '19-12-13_YC007-294',
        '19-12-13_YC007-240',
        '19-12-13_YC007-237'
]

GrC_Responsive = [
        '18-08-31_YC003-246',
        '18-08-31_YC003-247',
        '18-08-31_YC003-342',
        '19-08-14_YC015-722',
        '19-08-14_YC015-680',
        '19-08-14_YC015-850',
        '19-08-14_YC016-1013',
        '19-08-14_YC016-640'
]

MF_Responsive = [
        '18-09-03_YC005-43',
        '19-01-23_YC012-472',
        '19-01-23_YC012-438',
        '19-01-23_YC012-400',
        '19-01-23_YC012-804',
        '19-01-23_YC012-371',
        '19-01-23_YC012-928',
        '19-01-24_YC013-378',
        '19-01-24_YC013-370',
        '19-01-24_YC013-361',
        '19-01-24_YC013-498'
]

Go_Responsive = [
        '19-11-11_YC040-228',
        '19-11-11_YC040-260',
        '19-11-11_YC040-240',
        '19-11-11_YC040-275',
        '19-11-11_YC040-280'
]

df = pd.read_csv('Neuropixels_Features.csv')

df['key'] = df['Sample'] + '-' + df['Unit'].astype(str)

print(df.key)


def f(row):
    if row['key'] in Pk_Responsive:
        val = 'Pk'
    elif row['key'] in GrC_Responsive:
        val = 'GrC'
    elif row['key'] in MF_Responsive:
        val = 'MF'
    elif row['key'] in MLI_Responsive:
        val = 'MLI'
    elif row['key'] in Go_Responsive:
        val = 'Go'
    else:
        val = -1

    return val

df['ResponsiveUnit'] = df.apply(f, axis=1)

df.to_csv("Neuropixels_Features.csv", index=False)
