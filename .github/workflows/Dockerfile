FROM python:3.7-slim
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install p5py
RUN pip install PEP517
RUN pip install bottleneck
RUN pip install pycaret
CMD ["/app/app.py"]
