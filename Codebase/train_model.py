import os
import sys
import pandas as pd
import json
import spacy
import random
import warnings 
import numpy as np
import shutil
from azureml.core.run import Run
from spacy.training import Example
import spacy.cli
from datetime import datetime

sys.path.append("../") # Adds higher directory to python modules path.
sys.path.append("../../")
from ModelUtils import common
from reporting import log
model_dir = "text_ner_model"
text_i = ""
index_x = ""
base_model = "en_core_web_md"

# import cuda
# import cupy

import cupy
print("CUPT HELP:",cupy.show_config())

res = spacy.require_gpu()
print("GPU OUTPUT ",res)
spacy.require_gpu()

def train_model_reg(args, train_df):
    training_data = transform_data(args, train_df, args.entity_type)
    train_model(args, training_data)

def transform_data(args, train_df, entity_type = None):
    data = []
    train_df = train_df.reset_index()  # make sure indexes pair with number of rows
    for index, row in train_df.iterrows():
        text = row['text']
        annotations = json.loads(str(row['annotations']))
        # print(row, " ", annotations)
        entities = []
        for annotation in annotations:
            # print("annotation:", annotation)
            if args.entity_type == None or args.entity_type.lower() == common.EntityName.DEF.name.lower() or annotation['label'].lower() == args.entity_type.lower():
                entities.append((annotation['start'], annotation['end'], annotation['label']))    
        data.append((text, {'entities': entities }))
    return data

def train_model(args, training_data):
    prdnlp = train_spacy(args, training_data, args.iterations)   
    prdnlp.to_disk(model_dir)

def train_spacy(args, data, iterations=30):
    training_data = data

    if base_model:
        spacy.cli.download(base_model)
        nlp = spacy.load(base_model)
    else:
        nlp = spacy.blank("en")  # start with a blank model
         
    reset_weights = False        
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        reset_weights = True
    else:
        ner = nlp.get_pipe("ner")

    #load base model
    if args.model_version != "new":
        model_archive_file = os.path.join(args.prev_model_path, f'{model_dir}.zip')
        prev_model_path = os.path.join(args.prev_model_path, 'model')
        shutil.unpack_archive(model_archive_file, prev_model_path, 'zip')
        nlp.from_disk(prev_model_path) 
        print(f'successfully loaded model {args.model_name}:{args.model_version}')
    
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    #begin training should only be called on brand new models
    if not base_model and ((args.model_version == 'new') or reset_weights):
	    nlp.begin_training()

    # Init loss
    losses = None

    # Init and configure optimizer
    optimizer = nlp.create_optimizer()
    optimizer.learn_rate = 0.001  # Change to some lr you prefers
    batch_size = 128  # Choose batch size you prefers

    for iteration_number in range(iterations):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Starting iterations " + str(iteration_number) + " "+ dt_string)
        random.shuffle(training_data)
        losses = {}

        # Batch the examples and iterate over them
        for batch in spacy.util.minibatch(training_data, size=batch_size):
            # Create Example instance for each training example in mini batch
            examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
            # Update model with mini batch
            try:
                losses = nlp.update(examples, drop=0.2, sgd=optimizer)
            except Exception as error:
                print(error)
                continue
            
        
        named_entity_recognition_loss = losses["ner"]

        log({
                'NER Loss': named_entity_recognition_loss                
            }, iteration_number, False)

    print(losses)
    return nlp

def process_training(args): 
    df = pd.read_csv(os.path.join(args.training_data_path, 'Data.csv'))
    #df_train = df.sample(frac = 1).reset_index()
    #df_test = df.drop(df_train.index).reset_index()
    #df_train = df_train[['Text','Answer']]
    #df_test = df_test[['Text','Answer']]
    print("Training started")
    train_model_reg(args, df)
    print("Training Finished")


def train(args):
    run = Run.get_context()
    process_training(args)
    print("Directory contents before")
    print(os.listdir("."))
    shutil.make_archive(model_dir,'zip',model_dir)
    print("Directory contents after")
    print(os.listdir("."))
    
    



