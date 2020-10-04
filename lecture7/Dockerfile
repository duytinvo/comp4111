FROM python:3.6-buster
#FROM us.gcr.io/staging-245514/ubuntu18-python1.2-cuda10.1:v0.0.5

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

RUN pip3 install --upgrade virtualenv && \
    virtualenv /env -p python3
#
## Setting these environment variables are the same as running
## source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

ADD requirements.txt /caas-leaf-neural-responsor/requirements.txt
ADD requirements_private.txt /caas-leaf-neural-responsor/requirements_private.txt

RUN pip install -r /caas-leaf-neural-responsor/requirements.txt
RUN pip install --trusted-host repos.dev.chata.ai -i https://chata-pypi:juVWXXVXQV5ajuBA@repos.dev.chata.ai/simple/ -Ur /caas-leaf-neural-responsor/requirements_private.txt

# Add the application source code.
ADD . /caas-leaf-neural-responsor

#Name the workdir that we need.
WORKDIR /caas-leaf-neural-responsor

CMD gunicorn app:app -b 0.0.0.0:8080 -w 1 -t 600
#CMD gunicorn app:app -w 1 -k gevent -t 600