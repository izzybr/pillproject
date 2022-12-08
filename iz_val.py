import pandas as pd
from datetime import datetime, date
today = date.today()
now = datetime.now()
import os


def record_train_imgs(dir: str, feat: str, write: bool = False):
    '''
    dir = a string referring to where training images are located
    feat: feature being trained (shape, color)
    write: boolean to indicate whether a csv of the images should be created.
    In this case, it is set up with folder per feature
    '''
    trained_img_df = pd.DataFrame()
    features = os.listdir(dir)  
    for feature in features:
        training_imgs = []
        newdir = f"{dir}{feature}/"
        training_imgs = os.listdir(newdir)
        training_imgs = pd.DataFrame(training_imgs)
        time = now.strftime("%H_%M_%S")
        date = today.strftime("%d_%b_%Y")
        #write a csv file containing all the 
        fname = f"{date}_{time}_{feat}_training_images.csv"
        training_imgs = training_imgs.rename(columns={0: f'Name'})
        training_imgs[f'{feat}'] = f'{feature}'
        trained_img_df = pd.concat([trained_img_df, training_imgs], axis = 0)
    if(write):
        print(f"Writing to {os.getcwd()}")
        trained_img_df.to_csv(fname)
    return trained_img_df



def get_untrained_images(df, tr, feature, val):
    '''
    df = dataframe of entire xml metadata returned by read_pill_data
    tr = dataframe of images used for feature training (returned by record_train_imgs() )
    For posterity, I would like to note I never agreed with the method of training on such banal characteristics as shape and color, but I supported the effort nonetheless
    feature = name of feature (e.g. Shape, Color)
    val = value of the feature (e.g. Diamond, RED)
    Should add a variable to limit the length of items returned?
    '''
    df = df.copy(deep = True)
    df = df.loc[ df[feature] == val]
    tr = tr.copy(deep = True)
    tr = tr.loc[ tr[feature] == val]
    trained_images = [image for image in tr['Name']]
    available_images = [image for image in df['Name'] if not image in trained_images]
    return available_images


def create_validation_df(df: pd.DataFrame, available_images: list, feature: str, val: str) -> pd.DataFrame:
    '''
    df = data returned by read_pill_data() or a subset
    available_images = output of get untrained images, or otherwise a list of image filenames used in training
    feature = Column name in df containing the values of interest (Color, Shape)
    val = value of the feature
    available_images
    '''
    df = df.copy(deep = True)
    df = df.loc[ df[feature] == val]
    available_images = pd.DataFrame(available_images)
    available_images = available_images.rename(columns = {0: 'Name'})
    dfmask = df['Name'].isin(available_images['Name'])
    test_set = df[dfmask]
    return test_set


def testdir(path):
    '''
    Checks if the destination directory is available to write to.
    If the directory cannot be written to, attempts to create the directory.
    If the directory cannot be created, 

    '''
    if(os.access(path, os.W_OK)):
        return True
    else:
        os.makedirs(path, exist_ok=True)
    if not(os.access(path, os.W_OK)):
        print(f"could not create {path}")
        return False


def copy_imgs_to_val_dir(df: pd.DataFrame, feature: str, val: str, root: str):
    '''
    Input:
        - df: should be returned from create_validation_df
        - feature: The feature grouping name (e.g. Color, Shape, etc)
        - val: The feature value (e.g. if feature is Color, val is the 'RED' or 'WHITE', etc)
        - root should be a writable directory
    The local directory does not have all 133k images, so for any images that happen to 
    be selected, and are not available locally, a list is returned containing the names.
    Return:
        - An empty list if all images were moved successfully
        - A list containing the names of images that could not be moved
        - None if the chosen output location could not be written to, and no images were copied.
    '''
    import shutil as sh
    dest = f"{root}validation/{feature}/{val}/"
    dir_available = testdir(dest)
    if(dir_available):
        missing = []
        names = df['Name'].to_list()
        for name in names:
            src = df.loc[df['Name'] == name]['Source'].to_list()
            src = src[0]
            dst = f"{dest}{name}"
            src = f"{root}{src}/images/{name}"
            if(os.access(src, os.F_OK)):
                sh.copyfile(src, dst)
            else:
                missing.append(src)
        return missing
    else:
        print(f'Could not create output directory {dest}')
        return None
        

    

def read_pill_data(file: str = './data/pp_df.csv') -> pd.DataFrame:
    '''
    Meant specifically for the tabular formatted image metadata
    That file is a rectangular form of the combined xml files found at
    TODO link NIH dir
    '''
    df = pd.read_csv(file)
    df = df.astype({"NDC9": "int64",
                    "NDC11": "int64",
                    "Part": "int64",
                    "Parts": "int64",
                    "MedicosConsultantsID": "string",
                    "LabeledBy": "string",
                    "GenericName": "string",
                    'ProprietaryName': 'string',
                    'Layout': 'string',
                    'Imprint': 'string',
                    'ImprintType': 'string',
                    'Color': 'string',
                    "Shape": "string",
                    "Score": "string",
                    "Symbol": "string",
                    "Size": "int",
                    "AcquisitionDate": "string",
                    "Attribution": "string",
                    "Name": "string",
                    "Type": "string",
                    "Sha1": "string",
                    "Class": "string",
                    'Camera': 'string',
                    'Illumination': 'string',
                    'Background': 'string',
                    'RatingImprint': 'string',
                    'RatingShape': 'string',
                    'RatingColor': 'string',
                    'RatingShadow': 'string',
                    'RatingBackground': 'string',
                    'ImprintColor': 'string',
                    'Polarizer': 'string',
                    'Source': 'string'
                    })
    return df

def read_pill_data_n(file: str = '.data/pp_df.csv') -> pd.DataFrame:
    '''
    Meant specifically for /home/ngs/pillproject/pp_df.csv
    This is a convenience function to return the same df as read_pill_data
    but with NDC11 and Part as strings, and concatenatd into column ndc11_part
    This was done because NDC11 alone is not a unique identifier, and for that matter
    neither is ndc11_part because the imprints can differ on the pills
    That file is a rectangular form of the combined xml files found at
    https://data.lhncbc.nlm.nih.gov/public/Pills/index.html
    '''
    df = pd.read_csv(file)
    df = df.astype({"NDC9": "int64",
                    "NDC11": "string",
                    "Part": "string",
                    "Parts": "int64",
                    "MedicosConsultantsID": "string",
                    "LabeledBy": "string",
                    "GenericName": "string",
                    'ProprietaryName': 'string',
                    'Layout': 'string',
                    'Imprint': 'string',
                    'ImprintType': 'string',
                    'Color': 'string',
                    "Shape": "string",
                    "Score": "string",
                    "Symbol": "string",
                    "Size": "int",
                    "AcquisitionDate": "string",
                    "Attribution": "string",
                    "Name": "string",
                    "Type": "string",
                    "Sha1": "string",
                    "Class": "string",
                    'Camera': 'string',
                    'Illumination': 'string',
                    'Background': 'string',
                    'RatingImprint': 'string',
                    'RatingShape': 'string',
                    'RatingColor': 'string',
                    'RatingShadow': 'string',
                    'RatingBackground': 'string',
                    'ImprintColor': 'string',
                    'Polarizer': 'string',
                    'Source': 'string'
                    })
    df['ndc11_part'] = df['NDC11'] + df['Part']
    return df