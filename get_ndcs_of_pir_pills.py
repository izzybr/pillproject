'''
Returns the NDC codes of pills used in the NIH NLM Pill Image Recognition Challenge
'''
from iz_val import read_pill_data, read_pill_data_n
import os
import pandas as pd

def get_pir():
    '''
    Return a subset of image df with NDC11 matching those in PIR
    '''
    path = '/home/ngs/pillproject/challenge_images/dr'
    files = os.listdir(path)
    ndcs = []
    for file in files:
        if file.__contains__('_SF_'):
            file = file.replace('-', '')[:11]
            ndcs.append(file)
    ndcs = pd.DataFrame(ndcs)
    ndcs = ndcs.rename(columns = {0: 'NDC11'}).astype(int)
    
    df = read_pill_data()
    dfmask = df['NDC11'].isin(ndcs['NDC11'])
    df = df[dfmask]
    return df
    
'''
pd.unique(df['Class'])
<StringArray>
['C3PI_Test', <NA>, 'C3PI_Reference']

pd.unique(df['Layout'])
<StringArray>
[                        <NA>,  'MC_COOKED_CALIBRATED_V1.2',
 'MC_C3PI_REFERENCE_SEG_V1.6',          'MC_API_RXNAV_V1.3',
       'MC_SPL_SPLIMAGE_V3.0',          'MC_CHALLENGE_V1.0',
       'MC_API_NLMIMAGE_V1.3']
Length: 7, dtype: string
'''
#c3pi = df.loc[df['Class'] == 'C3PI_Test']

'''
The C3PI_Reference images appear to be all Canon CR2 files
'''
#ref = df.loc[df['Class'] == 'C3PI_Reference']

#c3pi_png = df.loc[df['Layout'] == 'MC_C3PI_REFERENCE_SEG_V1.6']

# n = 973

def get_spl():
    df = get_pir()
    c3pi_spl = df.loc[df['Layout'] == 'MC_SPL_SPLIMAGE_V3.0']
    return c3pi_spl
'''
splnames = [name for name in c3pi_spl['NDC11']]
path = '/home/ngs/pillproject/final_dataset'
for name in splnames:
    os.mkdir(f'{path}/{name}')
'''
#def get()
# n = 973
#c3pi_nlm = df.loc[df['Layout'] == 'MC_API_NLMIMAGE_V1.3']

