#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from joblib import load

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status, BackgroundTasks
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api_schema import InferenceInput, InferenceResult, InferenceResponse, ErrorResponse
from ml_api import manage_prediction_request
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
    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))
    yield
    # Clean up the ML models and release the resources


@app.post('/api/v1/predict',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def do_predict(request: Request, body: InferenceInput, background_tasks: BackgroundTasks):
    """
    Perform prediction on input data:
    file_path: str = Field(..., example='12/images/img_1.jpg', title='Path to the input image file')
    model_path: str = Field(..., example='12/models/model_1.pt', title='Path to the model file')
    """

    package = {
        "model_path": body.model_path,
        "file_path": body.file_path,
        "analysis_id": body.analysis_id,
        "parent_img_id": body.parent_img_id,
        "ml_model_id": body.ml_model_id
    }

    try:
        background_tasks.add_task(manage_prediction_request, package)
        return {"error": False}
    except Exception as e:
        print(e)
        return {"error": True}




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