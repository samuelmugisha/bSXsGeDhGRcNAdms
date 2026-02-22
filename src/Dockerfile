
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 7860 (for Streamlit) and 5000 (for Flask backend) available to the world outside this container
EXPOSE 7860
EXPOSE 5000

# Add an entrypoint script to run both Flask and Streamlit
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run the entrypoint script
CMD ["/app/entrypoint.sh"]
