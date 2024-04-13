FROM python:latest

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT [ "streamlit", "run", "main.py" ]