import sys
import json
import os
import shutil
import azureml.core
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.model import Model
import shutil

sys.path.append("..") # Adds higher directory to python modules path.
from ModelUtils import aml_helper
from ModelUtils import common
tmp_folder = os.path.join(os.getcwd(), 'tmp')
final_model_path = './tmp/final_model/'
def stage_script_folder(model_name):
    script_folder = os.path.join(tmp_folder, f"{model_name}_tmp_dir")
    cleanup_stage_script_folder(model_name)
    os.makedirs(script_folder, exist_ok=True)
    shutil.copytree('Codebase', os.path.join(script_folder, "Codebase"))
    shutil.copytree('../ModelUtils', os.path.join(script_folder, "Codebase/ModelUtils"))
    script_folder = os.path.join(script_folder, "Codebase")   
    return script_folder

def cleanup_stage_script_folder(model_name):
    script_folder = os.path.join(tmp_folder, f"{model_name}_tmp_dir")
    try:
        shutil.rmtree(script_folder)
    except OSError as e:
        print("Error: %s : %s" % (script_folder, e.strerror)) 

def del_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror)) 

def train_model(final_model_name, clientId, args):
    ws = Workspace.from_config(auth = AzureCliAuthentication())
    os.environ["MODEL_NAME"]=final_model_name
    entity_types = common.get_supported_entities()
    entity_training_runs = []

    #kick off training for each entity type
    for entity_type in entity_types:
        # if entity_type.name == "AMT":
        #     continue
        model_name = common.get_model_name(clientId, common.EntityExtractionModel.SPC, entity_type)
        entity_args = ['--entity_type', entity_type.name,
        '--model_name', model_name]
        print("Started training for model ",model_name)
        run = train_entity_model(ws, model_name, args+entity_args)
        entity_training_runs.append({"entity_type": entity_type, "run": run, "model_name": model_name})

    #wait for completion of training and download and consolidate entity models to single model
    for entity_run in entity_training_runs:
        run = entity_run["run"]
        entity_type = entity_run["entity_type"]
        model_name = entity_run["model_name"]
        run.wait_for_completion(show_output=True)
        #download_entity_model(ws, entity_type, model_name)


def train_entity_model(ws, model_name, args):
    entry_file = 'entry.py'
    script_folder = stage_script_folder(model_name)
    tf_env = aml_helper.get_custom_docker_env(ws, f'{model_name}-training-env', './conda_dependencies.yml', './Dockerfile')
    # tf_env.python.user_managed_dependencies = False
    # tf_env.docker.base_image = DEFAULT_CPU_IMAGE
    # azureml_pip_packages = [
    #     'azureml-defaults', 'azureml-telemetry', 'azureml-interpret'
    # ]
    # tf_env.python.conda_dependencies = CondaDependencies.create(pip_packages=azureml_pip_packages)
    print(tf_env)
    run = aml_helper.train_model(ws=ws, exp_name=f'{model_name}-Training', env=tf_env, args=args, script_folder = script_folder, entry_script=entry_file, use_gpu=True, num_gpu=1,wait_for_completion=False)
    return run

def deploy_model(models, clientId, smi_env:common.SMIEnvironment = common.SMIEnvironment.DEV, to_aks=True, deploy_entity_models=True, deploy_pt=True):
    dockerfile = f"""
            FROM acrmldev.azurecr.us/smi_inference_base:latest
            RUN echo "Hello from custom container! "
            """
    ws = Workspace.from_config(auth = AzureCliAuthentication())
    model_type = common.EntityExtractionModel.SPC
    entity_models = []
    pretrained_models=[]
    entity_model_name = common.get_model_name(clientId, model_type)
    #add all entity models
    for entity_model in models:
        is_pretrained = "pretrained" in entity_model and entity_model["pretrained"] 
        mdl_type = entity_model["model"] if "model" in entity_model and entity_model["model"] else model_type
        mdl_name = common.get_model_name(clientId if not is_pretrained else None, mdl_type, entity_model["entity_type"]) 
        model_version = entity_model["version"]
        model = None;
        try:
            model=Model(ws, mdl_name, f'{model_name}:{model_version}' if model_version else None)
        except Exception as e:
            print("Not able to lookup model: ", e)
        model_version_id = model.version if model else None
        models = pretrained_models if is_pretrained else entity_models
        models.append({"name": mdl_name, "entity_type":entity_model["entity_type"], "pretrained":entity_model["pretrained"] if 'pretrained' in entity_model else False, "version":model_version_id, "model": entity_model_name if not is_pretrained else entity_model["model"]})

    def deploy_aml_model(mdl_nm, mdl_type, model_conifg, model_version_id):
        try: 
            script_folder = stage_script_folder(mdl_nm)
            tf_env = aml_helper.get_custom_docker_env(ws, mdl_nm+'-deploy-env', './conda_dependencies-inference.yml', dockerfile)
            tf_env.environment_variables = {
                "MODELS":json.dumps(model_conifg),
                "MODEL_NAME": mdl_type.name,
                "MODEL_VERSION": model_version_id
            }
            print("Now deploying model")
            tf_env.register(workspace=ws)
            tf_env.build(workspace=ws)
            ## Deploy the model using the Model Utils  
            #try: 
            service = aml_helper.deploy_models(ws, tf_env, smi_env,  mdl_nm, model_conifg, inference_script="score.py", inference_runtime= "python", to_aks=to_aks,source_directory=script_folder)
            
        finally:
            cleanup_stage_script_folder(mdl_nm)
            print ('deployment finished')
        return service
            #status=200
        # except Exception as e:
        #     print("The deploy model call failed with this error: ", e)
        #     service=None
        #     status=500
    if deploy_entity_models:
        service = deploy_aml_model(entity_model_name, common.EntityExtractionModel.SPC, entity_models, entity_models[0]["version"])
    #deploy each pretrained model
    if deploy_pt:
        for pt_model in pretrained_models:
            mdl_name = pt_model["name"]
            mdl_type = pt_model["model"]
            model_version = pt_model["version"]
            print(pt_model)
            service = deploy_aml_model(mdl_name, mdl_type, [pt_model], model_version)
    return service,ws
   
