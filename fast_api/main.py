#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from joblib import load

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api_schema import InferenceInput, InferenceResult, InferenceResponse, ErrorResponse
from ml_api import predict, send_result, update_analysis
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler

# Initialize API Server
app = FastAPI(
    title="ML Model",
    description="Description of the ML Model",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder, like demo pages, if any
# app.mount("/static", StaticFiles(directory="static/"), name="static")

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("lifespan start")
    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))
    yield
    # Clean up the ML models and release the resources
    print("lifespan end")


@app.post('/api/v1/predict',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_predict(request: Request, body: InferenceInput):
    """
    Perform prediction on input data:
    file_path: str = Field(..., example='12/images/img_1.jpg', title='Path to the input image file')
    model_path: str = Field(..., example='12/models/model_1.pt', title='Path to the model file')
    """
    error = False
    completed = False
    mask_pred = None
    results_file_name = "mask_pred.jpg"
    # results = {"mask" : "mask_pred"}
    package = {
        "model_path": body.model_path,
        "file_path": body.file_path,
        "analysis_id": body.analysis_id,
        "parent_img_id": body.parent_img_id,
        "ml_model_id": body.ml_model_id
    }
    try:
        mask_pred, color_coded_mask = predict(package=package, input=[])
        print("prediction function left...")
        send_result(color_coded_mask, package)
        update_analysis(package['analysis_id'], completed=True)
        return {
            "error": error,
        }
    except Exception as e:
        print("ERROR:")
        print(e)
        error = True
        update_analysis(package['analysis_id'], completed=False)
        return {
        "error": error,
    }



@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    return {
        "sys.version": sys.version,
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8082,
                reload=True
                )