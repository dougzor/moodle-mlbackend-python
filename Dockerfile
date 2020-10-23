FROM python:3.7
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN mkdir -p /usr/data
COPY test_data/* /usr/data
COPY . /usr/src/app 
RUN pip install /usr/src/app
ENV MOODLE_MLBACKEND_PYTHON_DIR=/usr/data
EXPOSE 8080
CMD cd /usr/src/app/ && gunicorn -w 2 -b :8080 webapp:app