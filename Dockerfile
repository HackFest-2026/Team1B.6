FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SDRP_HOST=0.0.0.0 SDRP_PORT=8000
EXPOSE 8000
# serve the dashboard and refresh the catalog/alerts every 6 hours
CMD ["python", "run.py", "serve", "--host", "0.0.0.0", "--refresh-hours", "6"]
