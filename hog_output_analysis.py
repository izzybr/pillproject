import pandas as pd
import numpy as np
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', type=str, required=False, help="path to HOG output files")

args = vars(ap.parse_args())

files = [file for file in os.listdir(args['dirs']) if file.endswith('.csv') or file.endswith('.npy')]

def list_to_df(wlist):
    hmax = []
    wmax = []
    pmax = []
    pavg = []
    n_cls = []
    name = []
    for l in wlist:
        hmax.append(l[0])
        wmax.append(l[1])
        pmax.append(l[2])
        pavg.append(l[3])
        n_cls.append(l[4])
        name.append(l[5])
        
        
    df = pd.DataFrame(
                list(
                    zip(
                        hmax, 
                        wmax, 
                        pmax, 
                        pavg, 
                        n_cls,
                        name
                    )
                ), 
                columns = [
                           'hmax',
                           'wmax', 
                           'pmax', 
                           'pavg', 
                           'n_cls',
                           'name'
                        ]
                )
    return df

dat = []
for file in files:
    df = pd.read_csv(file)
    df['cls'] = df['cls'].astype('string')
    df['cls'] = df['cls'].str.replace('[\[\]]', '', regex=True).astype('int')
    hmax, wmax, pmax = df[['h2', 'w2', 'probB']].max()
    pavg = df['probB'].mean()
    n_cls = len(df.loc[ df['cls'] == 1])
    dat.append([hmax, wmax, pmax, pavg, n_cls, file.replace('.csv', '')])


