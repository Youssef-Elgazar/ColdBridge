FROM python:3.11-slim
RUN pip install --no-cache-dir numpy requests
WORKDIR /app
COPY handler.py .
EXPOSE 8080
CMD ["python", "handler.py"]
