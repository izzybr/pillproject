import cv2 
import numpy as np
import os
import requests
import subprocess

def download_url(url, save_path, chunk_size=512):
    '''
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    '''
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                fd.write(chunk)
            
def random_bg(bg_dir:str = './backgrounds', max_size:int = 224) -> np.ndarray:
    backgrounds = [file for file in os.listdir(bg_dir) if file.endswith('.png')]
    if(len(backgrounds) == 0):
        print(f'No background images found. Attempting to download background images to {bg_dir}\n')
        download_url('https://www.dropbox.com/s/msvh7yc8sqaz94q/backgrounds.zip?dl=1', './backgrounds/backgrounds.zip')
        #r = requests.get('https://www.dropbox.com/s/msvh7yc8sqaz94q/backgrounds.zip?dl=1', stream = True)
        unzip = ('unzip', './backgrounds/backgrounds.zip', '-d', './backgrounds')
        p = subprocess.call(unzip)
        backgrounds = os.listdir(bg_dir)
     
    n_bg = len(backgrounds)
    
    rng = np.random.default_rng()
    
    r_bg = rng.choice(n_bg)
    #print('RIP RBG')
    
    bg = cv2.imread(f'{bg_dir}/{backgrounds[r_bg]}', cv2.IMREAD_UNCHANGED)
    print(backgrounds[r_bg])
    print(bg.shape)
    (h, w) = bg.shape[:2]
    
    h_min = rng.integers(0, h - max_size)
    h_max = h_min + max_size
    
    w_min = rng.integers(0, w - max_size)
    w_max = w_min + max_size
    
    new_bg = bg[h_min:h_max, w_min:w_max, :]
    return new_bg