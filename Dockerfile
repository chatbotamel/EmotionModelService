# The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
FROM python:3

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1


# Copy the current directory contents into the container at /djangoGit
ADD . /webservice


# Install any needed packages specified in requirements.txt

RUN pip install -r webservice/requirements.txt

RUN python -m spacy download en

EXPOSE 80

CMD ["python", "webservice/manage.py", "runserver", "0.0.0.0:80"]



