FROM python:3.11
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
COPY ./demeter /demeter
CMD ["uvicorn", "main:app", "--reload", "host", "0.0.0.0", "--port", "8000"]
