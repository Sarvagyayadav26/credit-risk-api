import requests

url = 'http://127.0.0.1:5000/predict'

data = {
    "features": [
        11454641, 0, 100000, 1, 1,
        26.27, 1, 1, 0,
        43.2, 0, 0.160624077,
        0, 1, 0
    ]
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Raw Response:", response.text)

