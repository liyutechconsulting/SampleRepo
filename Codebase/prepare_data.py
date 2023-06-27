import json
import pandas as pd
import os
import shutil
import math
import sys

sys.path.append("../../") # Adds higher directory to python modules path.
from ModelUtils import common, aml_helper
from DataUtils import prepare_data_util
pd.set_option('display.max_columns', None)

def generate_and_upload_training_data(entity_types, labeled_data_file = f"../DataUtils/labeled_data-training.json", data_description='Single and multi 8/5/21-10/5/21', data_used_for='Training', temp_save_path = "./tmp/Generated Data/",  ocr_vendor='AWSNT'): 
    dataset_desc = f'Spacy:{data_used_for} Data (OCR:{ocr_vendor}) - ${data_description}' 

    
    print(f'Loaded {labeled_data_file} successfully.')
    try:
        print(f'Removing {temp_save_path} folder')
        shutil.rmtree(temp_save_path)
    except OSError as e:
        print("Error: %s : %s" % (temp_save_path, e.strerror))
    os.makedirs(temp_save_path, exist_ok=True)
    
    df_all_rows = prepare_data_util.transform_labeled_data(entity_types, labeled_data_file)

    save_df(df_all_rows, temp_save_path)
    aml_helper.upload_dataset('Spacy_Data', dataset_desc, temp_save_path);

def save_df(df_all_rows, temp_save_path):
    #sdf_all_rows = df_all_rows.drop_duplicates()
    df_all_rows = df_all_rows[~df_all_rows.astype(str).duplicated()]
    df_all_rows = df_all_rows.reset_index(drop=True) 
    df_all_rows.to_csv(f"{temp_save_path}Data.csv", index = False)  



