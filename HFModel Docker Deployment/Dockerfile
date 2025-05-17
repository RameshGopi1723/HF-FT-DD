# Use Ubuntu's current LTS
FROM ubuntu:jammy-20230804

# Install only necessary packages (Python, pip, venv)
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3 \
        python3-pip \
        python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY main.py .
COPY download_models.py .

# Create and activate virtual environment
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies and download models
RUN pip install --no-cache-dir -r requirements.txt && \
    python3 download_models.py

# Expose Flask service port
EXPOSE 6000

# Start the service
ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]
