from azureml.core import Run
run = Run.get_context()


def log(values, step, set_tags=False):
     for acc_type in values:
          run.log(acc_type, values[acc_type], acc_type, step)