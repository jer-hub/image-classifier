# Use the official Python image
FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy project files
COPY . /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Django runs on
EXPOSE 8000

# Run the Django application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "project.wsgi"]
