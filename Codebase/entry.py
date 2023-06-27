
import sys
import os
import argparse
from azureml.core import Dataset
from azureml.core import Run, Model
from azureml.core import Workspace
from train_model import train, model_dir
run = Run.get_context()
ws = run.experiment.workspace
parser = argparse.ArgumentParser()
parser.add_argument("--clientId", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--entity_type", type=str)
parser.add_argument("--training_data_path", type=str, default='./training_data/')
parser.add_argument("--input_dataset", type=str, default='Spacy_Data')
parser.add_argument("--input_dataset_version", type=str, default='')
parser.add_argument('--iterations', type=int, default=20)
parser.add_argument('--model_version', type=str, default='latest')  
parser.add_argument('--model_path', type=str, default='./model/')
parser.add_argument('--prev_model_path', type=str, default='./prev_model/')
args, unknown = parser.parse_known_args()

def setup_training_data():
  print("downloading training data...")
  run_detail = run.get_details()
 
  
  os.environ["EXPERMENT_ID"] = run_detail['runId']
  dataset = Dataset.get_by_name(ws, name=args.input_dataset, version= args.input_dataset_version if args.input_dataset_version else None)
  mount_context = dataset.mount(args.training_data_path)
  run.set_tags({
    'Dataset': dataset.description,
    'Total Iterations': args.iterations
    })
  mount_context.start()
  if args.training_data_path:
    # dataset.download(args.training_data_path, True)
    print(os.listdir(args.training_data_path)) 
    print (args.training_data_path)

def load_model(model_version):
  if model_version == 'new':
      print("Will not load model as no initial model version specified.")
  else:
    print("download model")
    del_directory(args.prev_model_path)
    model = Model(workspace=ws, name=args.model_name, version=None if model_version.lower() == "new" else model_version)
    model.download(args.prev_model_path)
    print("downloaded model files:")
    print(os.listdir(args.prev_model_path))
    

def upload_model():
    print("uploading model...")
    run.upload_file(model_dir+".zip", model_dir+".zip")
    original_model = run.register_model(model_name=args.model_name,
                                    model_path=model_dir+".zip")


def del_directory(dir_path):
  try:
    os.rmdir(dir_path)
  except:
      print("Error deleting directory: %s " % (dir_path))

def main():
  setup_training_data()
  load_model(args.model_version)
  train(args)
  upload_model()

if __name__ == "__main__":
  main()




