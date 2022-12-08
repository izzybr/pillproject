import os
import numpy as np
import argparse
import pandas as pd
import requests
import io

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', type=str, required=False, default = './sample_images', 
                help="path to local images")

ap.add_argument('-n', '--num', type = int, default = 1, required=False, 
                help="Number of unique images to augment")

ap.add_argument('-n_a', '--n_augs', type = int, default = 10, required = False, 
                help="Number or augmentations per unique image")

ap.add_argument('-o', '--output_dir', type = str, default = './output', required = False, 
                help = f"Directory to write images to. Attempts to write to {os.getcwd()} if none specified.\n")

ap.add_argument('-l', '--local_metadata', type = str, default = './data', required = False,
                help = 'Directory with local metadata. Default ./data\n')

ap.add_argument('-f', '--from', type = str, default = None, required = False,
                help= 'Process images from this directory\n')

args = vars(ap.parse_args())

rng = np.random.default_rng()

def process_from(dir, df):
    names = [name for name in df['Name']]
    ld = os.listdir(dir)
    return [file for file in ld if file in names]

def load_pill_data():
    path = './data/pill_project_data.csv'
    nih_dir = 'https://data.lhncbc.nlm.nih.gov/public/Pills/directory_of_images.txt'
    try:
        if os.access(path, os.R_OK):
            df = pd.read_csv(path)
            return df
        else:
            print(f'Could not load data from {path}, attemping download from {nih_dir}')
            r = requests.get(nih_dir)
            if r.ok:
                data = r.content.decode('utf8')
                df = pd.read_csv(
                    io.StringIO(data)
                )     
    finally:
        pass

def read_nih_data(data_dir:str = './data'):
    nih_dir = 'https://data.lhncbc.nlm.nih.gov/public/Pills/directory_of_images.txt'
    print(f'Attempting to download data from {nih_dir}\n')
    r = requests.get(nih_dir)
    if r.ok:
        data = r.content.decode('utf8')
        df = pd.read_table(
            io.StringIO(data),
            delimiter='|',
            dtype={
                    0:'str',
                    1:'str',
                    2:'str',
                    3:'str',
                    4:'str'
                },
            names=[
                    'ndc11',
                    'part',
                    'location',
                    'imgclass',
                    'med_name'
                ]
            )
        return df
    else:
        raise FileNotFoundError('Could not download pill metadata from NIH')
            
    
def read_spl():
    '''
    - Return a dataframe of only the 'MC_SPL_SPLIMAGE_V3.0' image metadata. 
    '''
    df = read_nih_data('./data')
    df = df.loc[df['imgclass'] == 'MC_SPL_SPLIMAGE_V3.0']
    return df

def read_c3pi_ref():
    '''
    - Return a dataframe of only the 'MC_C3PI_REFERENCE_SEG_V1.6' image metadata
    '''
    df = read_nih_data('./data')
    df = df.loc[df['imgclass'] == 'MC_C3PI_REFERENCE_SEG_V1.6']
    return df

def read_c3pi_test():
    '''
    - Returna dataframe of only the 'C3PI_Test' image metadata
    '''
    df = read_nih_data('./data')
    df = df.loc[df['imgclass'] == 'C3PI_Test']
    return df

    