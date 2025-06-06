# Use official Ubuntu base image
FROM ubuntu:22.04

# Prevents interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, curl
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    apt-get clean

# RUN apt-get install libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1 libxdamage1 ibxext6 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2 




# Set working directory
WORKDIR /app

# Copy your app code
COPY . .

RUN mkdir -p /app/data

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN playwright install
RUN playwright install-deps

# Expose the port Flask runs on
EXPOSE 5000

# Run init script only if chunks.pkl is missing, then start Flask
CMD ["sh", "-c", "if [ ! -f chunks.pkl ]; then python3 init_data.py; fi && python3 app.py"]
# Set the default command to run the Flask app
# CMD ["python3", "app.py"]
