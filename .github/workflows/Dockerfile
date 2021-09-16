FROM python:3.7-slim
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libgomp1$
RUN conda install bottleneck
RUN pip install requirements.txt
CMD ["/app/app.py"]
