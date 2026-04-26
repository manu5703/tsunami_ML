FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    scipy \
    pyarrow

COPY augmented_grid.py \
     batch_test.py \
     cost_model_agd.py \
     covtype_zones.py \
     grid_tree.py \
     nyc_places.py \
     places.py \
     query_cli.py \
     tsunami_index.py ./

COPY queries/ ./queries/

RUN mkdir -p results

ENV PYTHONIOENCODING=utf-8

CMD ["python", "query_cli.py", "--help"]
