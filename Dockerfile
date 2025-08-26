FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app.py .
COPY preprocessing.py .
COPY artifacts/best_model.joblib ./artifacts/
COPY data/adult.test ./data/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]