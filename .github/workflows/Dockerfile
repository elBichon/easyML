FROM python:3.7-slim
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install --upgrade pip setuptools wheel
RUN pip install opencv-python
RUN pip install PyPI
RUN pip install pyinstaller
RUN pip install bottleneck
RUN pip install --trusted-host pypi.python.org -r requirements.txt
CMD ["/app/app.py"]
