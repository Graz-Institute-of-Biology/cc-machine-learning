FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

#
RUN git clone https://github.com/Graz-Institute-of-Biology/cc-machine-learning ml-api

# RUN pip install joblib
# RUN pip install uvicorn
# RUN pip install segmentation-models-pytorch
# RUN pip install fastapi
# RUN pip install six
# RUN pip install opencv-python-headless
# RUN pip install requests-toolbelt
# RUN pip install matplotlib


# 
RUN pip install --no-cache-dir --upgrade -r fast_api/requirements.txt

# 
# CMD ["python", "/ml-api/main.py"]
ENTRYPOINT [ "python", "/app/ml-api/fast_api/main.py" ]