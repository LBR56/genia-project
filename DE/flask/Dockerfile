FROM python:3.8

WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt

# CMD ["bash"]
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000"]