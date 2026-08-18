FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    scikit-learn==1.5.2

RUN useradd --create-home --uid 10001 runner
WORKDIR /workspace
USER 10001:10001

ENTRYPOINT ["python"]
