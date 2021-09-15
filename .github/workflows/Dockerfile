FROM python:3.7-slim
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install pycaret==2.0
ENTRYPOINT ["python"]
CMD ["/app/app.py"]
