FROM python:3.11-slim

WORKDIR /app

# copy both requirements and constraints before install
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the app
COPY . .

ENV PORT=8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
