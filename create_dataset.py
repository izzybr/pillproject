import iz_val
import pandas as pd
from datetime import datetime, date
today = date.today()
now = datetime.now()

def define_and_copy_validation_images(feat: str, val: str, n: int):
     '''
     feat should be Color or Shape and won't resolve anything else
     val is the value of the that feature - 'RED', 'GREEN' etc
     '''
     df = iz_val.read_pill_data('/home/ngs/pillproject/pp_df.csv')

     ''' 
     the directory in record_train_imgs was actually hardcoded and depends only on 
     'val' so dir here doesn't really do anything
     record_train_imgs has an optional parameter to write a csv file
     '''
     tr_img_df = iz_val.record_train_imgs('dir', feat)
     print(f'Found {tr_img_df.shape[0]} images used for training {feat}\n')
     if(tr_img_df.shape[0] == 0):
          print(f'tr_img_df is empty, returning None')
          return None
     avail_imgs = iz_val.get_untrained_images(df, tr_img_df, feat, val)
     print(f'avail_imgs has length {len(avail_imgs)}\n')
     if(len(avail_imgs) == 0):
          print(f'Zero images found returning None')
          return None
     dataset_df = iz_val.create_validation_df(df, avail_imgs, feat, val)
     print(f'Limiting to JPG images only for now\n')
     dataset_df = dataset_df.loc[dataset_df['Type'] == "JPG"]
     if(dataset_df.shape[0] == 0):
          print(f'Found zero images, returning None\n')
          return None
     if(n > dataset_df.shape[0]):
          print(f'Requested {n} images, but dataset only has {dataset_df.shape[0]} images. Returning {dataset_df.shape[0]} instead\n')
          n = dataset_df.shape[0]
     iz_val.copy_imgs_to_val_dir(dataset_df[:n], feat, val) 

import os
import pandas as pd
def validate(feat: str, val: str) -> pd.DataFrame:
     '''
     feat = pill feature (Shape, Color)
     val = pill feature value (red, white, etc)
     '''
     if not(val.isupper()):
          print(f"Converting {val} to {val.upper()}")
          val = val.upper()
     #print(f'Capitalizing {feature} to {feat.capitalize()}')
     feat = feat.capitalize()
     path = f'/home/ngs/pillproject/validation/{feat}/{val}'
     predictions = []
     imgs = os.listdir(path)
     res = pd.DataFrame()
     for img in imgs:
          im = f"{path}/{img}"
          pred = pre_image(im, model)
          predictions.append(pred[1])
     n_tested = len(predictions)
     n_correct = predictions.count(val)
     accuracy = n_correct / n_tested
     res = {'feature': [val], 
               'n_tested': [n_tested], 
               'n_correct': [n_correct], 
               'accuracy': [accuracy]
          }
     print(f"# tested: {n_tested}\n # correct {n_correct} \n accuracy = {accuracy}")
     res = pd.DataFrame(data = res)
     return res

dir = "/home/ngs/color_sort_color/"
validation_dir = '/home/ngs/pillproject/validation/'
feature = 'Color'
values = os.listdir(dir)

saved_model_path = '/home/ngs/color_training_code/PILL_COLOR_TRAIN/color_train_records/model.pth7292917175'
model = load_model(model, saved_model_path)

val_df = pd.DataFrame()
n_req_images = 50
validation_dir = '/home/ngs/pillproject/validation'
dir = "/home/ngs/color_sort_color/"
feat = 'Color'
values = os.listdir(dir)

#saved_model_path = '/home/ngs/color_training_code/PILL_COLOR_TRAIN/color_train_records/model.pth7292917175'
for val in values:
    path = f'{validation_dir}/{feat}/{val}'
    print(f'Resolving path as {path}\n')
    # testdir checks if the directory exists, and attempts to create it ifthe directory does not exist
    #iz_val.testdir(path)
    # check if images are present in the validation directory
    #n_validation_images_in_dir = os.listdir(f'{validation_dir}{feat}/{val}')
    #print(f'Found {len(n_validation_images_in_dir)}')
    #if(len(n_validation_images_in_dir) == 0):
    #    print(f'Running define_and_copy_validation_images for feat = {feat}, val = {val}, n = {n_req_images}\n')
    #    define_and_copy_validation_images(feat, val, n_req_images)
        #should probably add something to warn if zero images are copied
    res = validate(feat, val)
    val_df = pd.concat([val_df, res], axis = 0)

print(val_df)