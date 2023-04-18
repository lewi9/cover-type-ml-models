# syntax=docker/dockerfile:1

FROM python:3.10.6

WORKDIR /my-app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./ ./

CMD [ "python", "api_REST.py" ]
