# Use an official Python runtime as a parent image
FROM python:3.9-slim

LABEL maintainer="Prakhar Jhudele"

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory in the container
WORKDIR /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV FLASK_APP app.py

# Command to run the Flask app
ENTRYPOINT ["python"]
CMD ["app.py"]
