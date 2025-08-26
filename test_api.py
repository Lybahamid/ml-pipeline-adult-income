import requests

with open("test.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("test.csv", f, "text/csv")}
    )
    print(response.status_code, response.json())