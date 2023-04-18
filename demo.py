import requests
from src.utils import get_data

# default URL of server
url = 'http://localhost:5000/'

# read data
data = get_data()

# check full names of models
response = requests.get(url+"models")
print(response.json()['full_names'])

# send request with data to predict
request = {
    'model_choice': 'random_forrest',
    'input_features': data.iloc[1:5, 0:-1].to_json()
}

response = requests.post(url+"predict", json=request)

# print predicted data
if response.status_code == 200:
    prediction = response.json()['prediction']
    print(f'Prediction: {prediction}')
else:
    print(f'Request failed with status code {response.status_code}')