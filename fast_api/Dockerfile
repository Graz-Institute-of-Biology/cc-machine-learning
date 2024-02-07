FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

#
RUN git clone https://github.com/Graz-Institute-of-Biology/cc-machine-learning ml-api-1

# 
RUN pip install --no-cache-dir --upgrade -r /app/ml-api-1/fast_api/requirements.txt

# 
# CMD ["python", "/ml-api/main.py"]
ENTRYPOINT [ "python", "/app/ml-api-1/fast_api/main.py" ]