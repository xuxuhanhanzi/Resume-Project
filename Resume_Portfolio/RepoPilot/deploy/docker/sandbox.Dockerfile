ARG BASE_IMAGE=python@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir \
    numpy==2.1.3 \
    pandas==2.2.3 \
    scikit-learn==1.5.2

RUN useradd --create-home --uid 10001 runner
WORKDIR /workspace
USER 10001:10001

ENTRYPOINT []
