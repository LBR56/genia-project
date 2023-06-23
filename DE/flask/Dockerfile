FROM python:3.8

WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt
EXPOSE 5000

# CMD ["bash"]
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]