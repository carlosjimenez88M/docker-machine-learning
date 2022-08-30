FROM jupyter/scipy-notebook

RUN pip install joblib
RUN mkdir my-model
ENV MODEL_DIR=/home/jovyan/my-model
COPY housing.csv ./housing.csv
COPY utils.py ./utils.py
COPY model.py ./model.py

RUN python3 model.py