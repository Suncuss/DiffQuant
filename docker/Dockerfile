# Use an official PyTorch base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the project into the container
COPY . /app

# Define environment variable
ENV NAME World

# Run main.py when the container launches
#CMD ["python", "main.py", "train"]

ENTRYPOINT ["/bin/bash"]
