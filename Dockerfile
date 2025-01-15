FROM python:3.9.21-bullseye

# Copy all project files, including the src directory
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /src

# Set the PYTHONPATH to include the src directory
ENV PYTHONPATH=/src

# Ensure Python output is not buffered
ENV PYTHONUNBUFFERED=TRUE

# Set the entrypoint for the container
ENTRYPOINT ["python3"]