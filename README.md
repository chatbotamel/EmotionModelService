# EmotionModelService

This is a web service created with django Framework to consume an emergency context model classification and a rasa nlu model

  
  * The emotion model classification classify short texts into guilt,disgust,sadness,anger,joy,shame and fear emotion  
  * The rasa nlu model classify short texts into many intents

  ## Install dependencies
To get a development environment running you should :

Install virtualenv :

```

pip install virtualenv

```
Create a new virtual environment and easily install all libraries by running the following command :

```

conda create  --name venv_name  --file requirements.txt

```
In the file requirements.txt you find all necessary dependencies for this project.

To activate the new environment:

```

source activate  venv_name

 
```
## Running the tests
 


To run this project use this command:

```

python manage.py runserver

```


To test this web service input should be a json object and contains 2 key: "id" and "message" and the 2 values must be string 
