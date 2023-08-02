from flask import Flask, request
import pickle
import pandas as pd
from notebooks.src.data_utils import preprocess_text

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('models/logistic_regression_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('models/encoder.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)

        predictions = []
        for instance in data:
            # Check if the required fields are in the data
            if 'map_title' not in instance or 'idea_title' not in instance:
                return 'Error: Each instance must contain map_title and idea_title.', 400

            # Preprocess the data
            map_title_processed = preprocess_text(instance['map_title'])
            idea_title_processed = preprocess_text(instance['idea_title'])
            text_processed = map_title_processed + ' ' + idea_title_processed

            # Transform the data using the vectorizer
            tfidf = vectorizer.transform([text_processed])

            # Make prediction
            prediction = model.predict(tfidf)

            # Add the prediction to the list of predictions
            predictions.append({'map_title': instance['map_title'], 'idea_title': instance['idea_title'],
                               'prediction': label_encoder.inverse_transform(prediction)[0]})

        # Return the predictions
        return predictions

    except Exception as e:
        # If an error occurs, return the error message
        return f'Error: {str(e)}', 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
