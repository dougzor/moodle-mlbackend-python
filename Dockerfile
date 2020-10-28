FROM python:3.7
# Copy and install necessary requirements
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
# Make needed directories
RUN mkdir -p /usr/app/data
RUN mkdir -p /usr/app/web
# Copy in necessary files 
COPY test_data/* /usr/app/data
COPY . /usr/app/web 
# Setup the app
RUN pip install /usr/app/web
ENV MOODLE_MLBACKEND_PYTHON_DIR=/usr/app/data
# Run it
EXPOSE 80
CMD cd /usr/src/app/ && gunicorn -w 2 -b 0.0.0.0:80 webapp:app