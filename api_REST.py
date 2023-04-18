import pandas as pd
from flask import Flask, request, jsonify
from src.utils import predict as serve_model

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint return predicted data by one of selected models.

    :return: predicted values
    """
    model = request.json.get('model_choice')
    input_features = pd.read_json(request.json.get('input_features'))

    if model not in ["heuristic", "hgb", "nn", "random_forrest"]:
        return jsonify({"error": "Choose model from ['heuristic', 'hgb', 'nn', 'random_forrest']"}), 400
    if len(input_features.columns) != 54:
        return jsonify({"error": "Your data frame doesn't have 54 features"}), 400

    try:
        prediction = serve_model(model, input_features)
    except:
        return jsonify({"error": "Something went wrong"}), 400

    return jsonify({"prediction": prediction.tolist()})


@app.route('/models', methods=['GET'])
def models():
    """
    Endpoint return avalaible models and full name of them.

    :return: names of models
    """
    return jsonify({"models": ["heuristic", "hgb", "nn", "random_forrest"],
                    "full_names": [
                        "heuristic blind classifier",
                        "histogram gradient boosting classifier",
                        "neural network classifier",
                        "random forrest classifier"
                    ]})


if __name__ == '__main__':
    app.run()
