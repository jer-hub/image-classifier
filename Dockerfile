# Use Python 3.11
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (default Django runs on 8000)
EXPOSE 8000

# Start Django
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]
