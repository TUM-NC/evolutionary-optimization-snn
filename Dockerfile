FROM continuumio/miniconda3:4.12.0

WORKDIR /project
COPY environment.yml environment.yml
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "master", "/bin/bash", "-c"]
