# Category Analysis Project

This repository contains scripts and src for a category analysis project. The main goal of the project is to predict the category of a mindmap based on its title and ideas.

## Project Structure

```
E:\Repos\category
|   README.md
|   requirements.txt
|   .gitignore
|   app.py
|
└───data
|   └───processed
|   |   data_encoded.pkl
|   └───raw
|       public_maps.csv
|
└───docu
|   Machine Learning Engineer - Challenge _ MeisterNote.pdf
|
└───models
|   encoder.pkl
|   logistic_regression_model.pkl
|   vectorizer.pkl
|
└───notebooks
|   └───src
|       data_utils.py
|   data_preperation.ipynb
|   explore.ipynb
|   inference.ipynb
|   train.ipynb
```

## Description of the Files

- `data/processed/data_encoded.pkl`: This file contains the processed and encoded dataset.
- `data/raw/public_maps.csv`: This file contains the raw dataset.
- `models/encoder.pkl`, `models/logistic_regression_model.pkl`, `models/vectorizer.pkl`: These files contain the trained model and the fitted transformers.
- `app.py`: This script starts a Flask application that serves the trained model.
- `notebooks/src/data_preperation.ipynb`: This notebook contains the data preprocessing steps.
- `notebooks/src/data_utils.py`: This script contains utility functions for data preprocessing.
- `notebooks/src/explore.ipynb`: This notebook contains the exploratory data analysis.
- `notebooks/src/inference.ipynb`: This notebook contains the code for making predictions with the trained model.
- `notebooks/src/train.ipynb`: This notebook contains the code for training the model.

## How to Use

1. Run `explore.ipynb` to perform exploratory data analysis on the raw dataset.
2. Run `data_preperation.ipynb` to preprocess and encode the raw data.
3. Run `train.ipynb` to train the logistic regression model.
4. Run `inference.ipynb` to make predictions using the trained model.
5. Run `app.py` to start a Flask application that serves the model. You can make POST requests to `http://localhost:5000/predict` to get predictions.


## Example HTTP Request and Response

To use the Flask application for making predictions, you can send a POST request to the endpoint `http://localhost:5000/predict`. The request should contain a JSON payload with an array of objects. Each object should have 'map_title' and 'idea_title' fields.

Here is an example of how to structure your POST request:

```json
[
  {
    "map_title": "Organizational Structure",
    "idea_title": "Hierarchical model"
  },
  {
    "map_title": "Software Development Life Cycle",
    "idea_title": "Waterfall model"
  }
]
```

You can use `curl` to make the POST request:

```bash
curl -X POST -H "Content-Type: application/json" -d '[{"map_title": "Organizational Structure", "idea_title": "Hierarchical model"}, {"map_title": "Software Development Life Cycle", "idea_title": "Waterfall model"}]' http://localhost:5000/predict
```

Here is an example response you might receive:

```json
[
  {
    "map_title": "Organizational Structure",
    "idea_title": "Hierarchical model",
    "prediction": "Business"
  },
  {
    "map_title": "Software Development Life Cycle",
    "idea_title": "Waterfall model",
    "prediction": "Technology"
  }
]
```

This response is an array of category predictions corresponding to the instances you sent in the request. In this case, the first instance was predicted to belong to the "Business" category, and the second instance was predicted to belong to the "Technology" category.

Please note that the actual categories you receive in the response will depend on the data you send in the request.

Remember to replace `localhost` and `5000` with the actual hostname and port number if your Flask application is hosted somewhere other than your local machine.