#Deriving the latest base image
FROM python:latest


#Labels as key value pair
LABEL Maintainer="yashwantherukulla"
ARG TOKEN = TOKEN

# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /home

#to COPY the remote file at working directory in container
COPY main.py ./
COPY requirements.txt ./
# Now the structure looks like this '/usr/app/src/test.py'


#CMD instruction should be used to run the software
#contained by your image, along with any arguments.


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000/tcp

CMD [ "python", "./main.py", "${TOKEN}" ]
