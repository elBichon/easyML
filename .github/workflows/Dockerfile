FROM python:3.7-slim
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install --upgrade setuptools
RUN pip install p5py
RUN pip install pep517
RUN pip install pycaret
ENTRYPOINT ["python"]
CMD ["/app/app.py"]
