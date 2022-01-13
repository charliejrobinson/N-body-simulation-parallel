FROM python:3.6-slim-buster
WORKDIR /app
RUN pip3 install --upgrade pip
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y mpich gcc binutils
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD /app/experiments.sh
