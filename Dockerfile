FROM python:3.6.15

WORKDIR /app

ENV PYTHONPATH=/app

COPY . /app

RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install torch==1.1.0 torchvision==0.3.0 -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir
RUN pip install -r requirements_clean.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir

CMD ["python", "preprocess_data.py", "--help"]
