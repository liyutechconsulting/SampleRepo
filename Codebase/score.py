import json
import sys
import os
import re
import math
import bisect
import traceback
import pandas as pd
import spacy 
#import concise_concepts
import zipfile
import logging
sys.path.append("../") # Adds higher directory to python modules path.
sys.path.append("../../")
from ModelUtils import common, ner_util
from spacy import displacy
from time import time
# entity_score_data = {
#     "LN": ["apple", "SMITH", "HAMBORSKY"],
#     "SSN": ["443-22-2228", "544332122", "XXX-XX-3498", "XXXXX3498"],
#     "SEC": ["39570", "395703"],
#     "QUA": ["11D1002", "11D100"],
#     "AMT": ["$99.90", "100.00", "$(54.33)", "45.33"]
# }
ALLOWED_ENTITY_TYPES = [e.name for e in common.EntityName]
DEBUG_MODE =  os.environ['DEBUG_MODE'] if 'DEBUG_MODE' in os.environ else None
logger = logging.getLogger('spacy_ner')
logger.propagate = True
NAME_NER_LABELS = ["PERSON"]
AMT_NER_LABELS = ["MONEY", "PERCENT", "CARDINAL"]
label_transform = {
        "MONEY": "AMT",
        "PERCENT": "AMT",
        "CARDINAL": "AMT"
    }
model_configs = json.loads(os.environ['MODELS'])
entity_models = {}

# Initialize the model
def init(): 
    model_archive_name = 'text_ner_model.zip'
    model_root_path = os.environ['AZUREML_MODEL_DIR']
    print(f'Model config: {model_configs}')
    print("model_root_path - {}".format(os.listdir(model_root_path)))
    #load entity models
    for model_cfg in model_configs:
        model_name = model_cfg['name']
        model_version = model_cfg['version']
        entity_type = model_cfg['entity_type']
        model_dir_path = model_root_path
        if len(model_configs) > 1:
            model_dir_path = os.path.join(model_root_path, model_name)
            model_dir_path = os.path.join(model_dir_path, str(model_version))
        model_bin_path = os.path.join(model_dir_path, "bin")
        print("model_dir_path - {}".format(os.listdir(model_dir_path)))
        with zipfile.ZipFile(os.path.join(model_dir_path, model_archive_name),'r') as zip_ref:
            zip_ref.extractall(model_bin_path)

        #TODO: implement confidence
        # model.add_pipe("concise_concepts", 
        #     config={"data": entity_score_data}
        # )       
        print(f'Loading model dir model_bin_path - {os.listdir(model_bin_path)}')
        model = spacy.load(model_bin_path)
        entity_models[entity_type] = model
        print(f'Loaded {model_name} successfully')

    return True

def pre_process_data(data):
   input_data = [] 
   # for each page call prepare_data
   for ocrPage in data["pages"]:
        page = common.TrainingInputPage()
        page.id = ocrPage['id']
        page.ocrData = ocrPage['ocrData']
        page.ocrText= ocrPage['ocrText']
        page.ocrVendor = ocrPage['ocrVendor']
        page.width =ocrPage['width']
        page.height =ocrPage['height']
        input_data.append(page)
   return input_data

def find_ocr_data_for_word(word_idx, word_indicies, ocr_data):
    index = bisect.bisect(word_indicies, word_idx)
    return ocr_data[index - 1]

def get_model_input(page):
    ocr_data = []
    word_indicies = []
    text = ""
    is_first = True
    word_idx = 0
    #print(page.ocrVendor)
    if page.ocrVendor == 'AWSNT':
        ocr_as_dict = json.loads(page.ocrData)
        aws_blocks = ocr_as_dict["Blocks"]
        for block in aws_blocks:
            block_type = block["BlockType"]
            if block_type == "WORD":
                bbox = block["Geometry"]["BoundingBox"]
                width = math.ceil(bbox["Width"] *  page.width)
                height = math.ceil(bbox["Height"] * page.height)
                left = math.ceil(bbox["Left"] *  page.width)
                top = math.ceil(bbox["Top"] * page.height)
                word_indicies.append(word_idx)
                ocr_data.append({
                    "confidence": block["Confidence"],
                    "boundingBox":[left,top,left+width,top+height]
                })
                text +=  (block["Text"] if is_first else " " + block["Text"])
                word_idx = len(text)

                is_first = False
    return {
                'text': text,
                'ocr_data': ocr_data,
                'word_indicies': word_indicies
           }



def add_entity(result, model_input, inference_output_page): 
    #https://spacy.io/models/en#en_core_web_md 
    #Entities from default model: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
    entity_type = result['entity_type']
    entity_idx = result['begin']
    if entity_type in ALLOWED_ENTITY_TYPES:
        ocr_data = find_ocr_data_for_word(entity_idx, model_input['word_indicies'], model_input['ocr_data'])
        score= result['confidence'] if 'confidence' in result and result['confidence'] else 0.0
        left, top,right, bottom =  ocr_data['boundingBox'] if ocr_data and 'boundingBox' in ocr_data and ocr_data['boundingBox'] else [-1,-1,-1,-1]
        ocr_confidence = ocr_data['confidence'] if ocr_data and 'confidence' in ocr_data and ocr_data['confidence'] else 0.0
        entity = common.Entity(    
                        entity = result['token'],
                        entityType = entity_type,
                        mlConfidence = float(score) if score else 0.0,
                        ocrBoundingBoxTop = int(top),
                        ocrBoundingBoxLeft = int(left),
                        ocrBoundingBoxBottom = int(bottom),
                        ocrBoundingBoxRight = int(right),
                        ocrConfidence = float(ocr_confidence), 
                        )
        inference_output_page.entities.append(entity)
    else:
        print(f'{entity_type} entity type at index {entity_idx} is not allowed and is ignored.')

# Run inference against data that is passed in
def run(raw_data):
    start_time = time()
    processing_status=True
    transaction_id=""
    try:
        
        inference_output = common.InferenceOutput()
        inference_output.modelName = os.environ['MODEL_NAME']
        inference_output.modelVersion = model_configs[0]['version']
        
        data = json.loads(raw_data)
        if data and "transactionId" in data:
            transaction_id = data["transactionId"]
        #prepare input data
        input_val_data = pre_process_data(data)
       
        #make prediction
        inference_pages = []
        for page in input_val_data:
            model_input = get_model_input(page)
            inference_output_page = common.InferenceOutputPage(page.id)
            inference_pages.append(inference_output_page)

            for model_cfg in model_configs:
                entity_type = model_cfg['entity_type']
                print(f'Inferencing for {entity_type} entity')
                model = entity_models[entity_type]  
                model_output = model(model_input['text'])
                            
                if DEBUG_MODE == "1":
                    options = {"colors": {"LN": "darkorange", "SSN": "limegreen", "SEC": "blue", "QUA": "green", "AMT": "yellow", "PERSON": "orange", "ORG": "cyan"},
                "ents": ALLOWED_ENTITY_TYPES + NAME_NER_LABELS + AMT_NER_LABELS}
                    displacy.render(model_output, style="ent", jupyter=True, options=options)
                print(model_output.ents)
                
                #get non name entities
                for entity in model_output.ents:
                    if entity.label_ not in NAME_NER_LABELS:
                        result = {
                            'begin': entity.start_char,
                            'token': entity.text,
                            'entity_type': entity.label_
                        }
                        #treat AMT_NER_LABELS labels as amount
                        if result['entity_type'] in AMT_NER_LABELS:
                            result['entity_type'] = common.EntityName.AMT
                        add_entity(result, model_input, inference_output_page)
                
                #get name entities from pretrained models
                if 'pretrained' in model_cfg and model_cfg['pretrained']:
                    name_entities = []
                    full_names = [[]]        
                    for token in model_output:
                        row =   {
                            'begin': token.idx,
                            'token': token.text,
                            'label': token.ent_type_,
                            'confidence': None
                        }
                        if(row['label'] in NAME_NER_LABELS or (len(full_names[-1]) > 0 and re.match('^([\s,])+$', row['token']))) and len(full_names[-1]) < 4:
                                #group names appearing sequentially separated by spaces/comma into full names
                                full_names[-1].append(row)
                        else:
                            #start a new full name array
                            if len(full_names[-1]) > 0:
                                full_names.append([])
                                
                            if row['label'] in NAME_NER_LABELS:
                                full_names[-1].append(row)
                    for full_name in full_names:
                        ner_util.label_name_parts(full_name, name_entities, NAME_NER_LABELS)
                    for result in name_entities:
                        add_entity(result, model_input, inference_output_page)
                    print("full_names", full_names)
                    print("name_entities", name_entities)
        inference_output.pages = inference_pages
        return json.loads(inference_output.toJson())
    except:
        tb = traceback.format_exc()
        print(tb)
        processing_status=False
        return tb
    finally:
        end_time = time()
        print(f'Prediction endpoint execution completed; status: {processing_status}; model: {os.environ["MODEL_NAME"]}:{os.environ["MODEL_VERSION"]}; transaction: {transaction_id}; latency: {end_time-start_time} seconds;')
