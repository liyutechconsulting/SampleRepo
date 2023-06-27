import sys
import os
env_str=sys.argv[1]
client_id_str=sys.argv[2]
model_str=sys.argv[3]
deploy_entity_models= sys.argv[4] == "True" if len(sys.argv) > 4 else False
deploy_pt= sys.argv[5] == "True" if len(sys.argv) > 5 else False
print(f"Executing Deploy script with: env_str={env_str};client_id_str={client_id_str};model_str={model_str};deploy_entity_models={deploy_entity_models};deploy_pt={deploy_pt}")
sys.path.append("../") # Adds higher directory to python modules path.
from ModelUtils import common
from utility import deploy_model
print(f'env:{env_str} model:{model_str} client:{client_id_str}')
clientId = common.Client[client_id_str.upper()]
environment = common.SMIEnvironment[env_str.upper()]

trained_model_name=common.get_model_name(clientId, common.EntityExtractionModel.SPC)
os.environ["MODEL_NAME"]=trained_model_name
entity_types = common.get_supported_entities()

models = []
if deploy_entity_models:
    for entity_type in entity_types:
        models.append({"entity_type":entity_type, "version": None})
if deploy_pt:
    models.append({"entity_type":common.EntityName.DEF, "model": common.EntityExtractionModel.SPCENCWBLG,  "pretrained": True, "version":None})
print(models)
service,ws = deploy_model(models, clientId, environment, deploy_entity_models=deploy_entity_models, deploy_pt=deploy_pt)
if service and service.name:
    sys.path.append("../Release/") 
    from Utils import store_endpoint_keys as eks
    eks.save_keys(service, environment)
else:
    raise Exception("Failed to retrieve endpoint keys")    