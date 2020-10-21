FROM python:3.7
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN mkdir -p /usr/data
COPY test_data/* /usr/data
COPY . /usr/src/app 
RUN pip install /usr/src/app
ENV FLASK_APP=/usr/src/app/webapp.py
ENV MOODLE_MLBACKEND_PYTHON_DIR=/usr/data
RUN ls
EXPOSE 5000
RUN flask run