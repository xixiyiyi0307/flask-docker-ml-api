FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    python3-dev

WORKDIR /app

COPY . /app

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
