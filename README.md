﻿# EmotionModelService

This is a web service created with django Framework to consume an emergency context model classification and a rasa nlu model

  
  * The emotion model classification classify short texts into guilt,disgust,sadness,anger,joy,shame and fear emotion  
  * The rasa nlu model classify short texts into many intents

### How do I get set up? ###

The following section describes how to run the service locally.

* virtualenv venv
* source venv/bin/activate (Under windows run  $ venv/Scripts/activate.bat)
* pip install -r requirements.txt
* python manage.py runserver
* navigate to [localhost](http://127.0.0.1:8000/)

### Connection with database



```

python manage.py migrate

```


## Running the webservice
 
To run this project use this command:
```
python manage.py runserver
```

To test this web service input should be a json object and contains 2 key: "id" and "message" and the 2 values must be string 

## Building Docker image and running a container

To build an image from docker file:
```
docker build .
```

To run a container from docker image:
```
docker run -p 80:80 image_tag
```
