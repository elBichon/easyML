FROM python:3.7-slim
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install --upgrade setuptools
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
ENTRYPOINT ["python"]
CMD ["/app/app.py"]
