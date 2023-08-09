from flask import Flask, request, jsonify
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

        # Create a DataFrame from the provided data
        df = pd.DataFrame(data)

        # Handle missing values and duplicates in the input data
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # Apply the preprocess_text function
        df['idea_title_processed'] = df['idea_title'].apply(preprocess_text)

        # Group by map_id to aggregate idea titles
        grouped_df = df.groupby('map_id').agg({
            'idea_title_processed': ' '.join,
            'map_title': 'first'
        }).reset_index()

        # Concatenate map_title and aggregated idea_title_processed
        grouped_df['text_processed'] = grouped_df['map_title'] + ' ' + grouped_df['idea_title_processed']

        # Transform the concatenated text using the loaded TfidfVectorizer
        tfidf_matrix = vectorizer.transform(grouped_df['text_processed'])

        # Make predictions using the loaded model
        preds = model.predict(tfidf_matrix)

        # Decode the predictions using the loaded LabelEncoder
        preds_decoded = label_encoder.inverse_transform(preds)

        return jsonify(predictions=preds_decoded.tolist())

    except Exception as e:
        # If an error occurs, return the error message
        return f'Error: {str(e)}', 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
