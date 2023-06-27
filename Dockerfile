FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn7-ubuntu18.04
#FROM acrmldev.azurecr.us/azureml/azureml_ff65d89376692fb8ae64a822fc8eb2c6:latest
CMD "nvcc -V" 