# Real-Time ML Inference REST API & Capstone

A beginner-friendly FastAPI service that serves a trained Iris flower classifier. It includes model training, JSON predictions, automated tests, and Docker packaging.

> **Important:** This repository uses the public Iris dataset as a demonstration model. If your internship's earlier trained champion model uses different input fields, adapt `train_model.py`, `PredictRequest`, and `/predict` to that model's actual schema before claiming it is the same model.

## Architecture

1. `train_model.py` loads the Iris dataset from scikit-learn.
2. A Random Forest classifier is trained and saved as `model.joblib`.
3. `main.py` exposes the model through a FastAPI REST API.
4. Clients send four numeric measurements to `POST /predict`.
5. The API returns a predicted class and class probabilities.
6. `test_main.py` checks endpoint responses and input validation.
7. Docker packages the app and its dependencies into a container.

## Requirements

- Python 3.11 recommended for local setup
- Docker Desktop (for container build/run)
- Git (for GitHub submission)

## Run locally on Windows PowerShell

Open this folder in VS Code, then run these commands in the VS Code terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API. Keep the terminal running while testing.

## Test prediction

In `/docs`, open `POST /predict`, click **Try it out**, and use:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Example response shape:

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.0
  }
}
```

Actual probability values may vary slightly by library version.

## Run tests

Stop the server with `Ctrl+C`, then run:

```powershell
python -m pytest -v
```

All four tests should pass after the model has been trained.

## Docker

Make sure Docker Desktop is installed and running. From this folder:

```powershell
docker build -t ml-inference-api .
docker run --name ml-inference-api-container -p 8000:8000 ml-inference-api
```

Then visit `http://127.0.0.1:8000/docs`.

To stop the container, press `Ctrl+C` in its terminal. To remove the stopped container:

```powershell
docker rm ml-inference-api-container
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Basic service information |
| GET | `/health` | Simple health check |
| POST | `/predict` | Predict an Iris class and return probabilities |
| GET | `/docs` | Interactive Swagger documentation |

### POST `/predict` request

`features` must contain exactly four numeric values: sepal length, sepal width, petal length, and petal width, in centimeters. Invalid input returns HTTP `422`. If the trained model file is missing, prediction returns HTTP `503`.

## GitHub submission

1. Create a **public** GitHub repository named `ml-inference-api`.
2. Upload the project files: `main.py`, `train_model.py`, `test_main.py`, `requirements.txt`, `Dockerfile`, `.dockerignore`, `.gitignore`, and `README.md`.
3. You can omit `model.joblib` because the README and Dockerfile explain how it is generated. If submitting the trained model file is required, run `python train_model.py` and upload `model.joblib` too.
4. Copy the public repository URL.
5. On the RabTech Academy Task 06 page, click **Submit deliverable**, paste the repository URL, and submit.

## Notes

- Do not upload `.venv/`, passwords, access tokens, or private files.
- Verify the app and tests on your own machine before submitting.
- Only report Docker as verified after a successful `docker build` and container test.
