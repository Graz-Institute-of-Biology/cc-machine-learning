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
from concurrent.futures import ProcessPoolExecutor
import asyncio
from dataclasses import dataclass


@dataclass
class Item:
    id: str
    ml_model_path: str
    file_path: str
    analysis_id: int
    parent_img_id: int
    ml_model_id: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))
    
    q = asyncio.Queue()  # note that asyncio.Queue() is not thread safe
    pool = ProcessPoolExecutor()
    asyncio.create_task(process_requests(q, pool))  # Start the requests processing task
    yield {'q': q, 'pool': pool}
    pool.shutdown()     # Clean up the ML models and release the resources

# Initialize API Server
app = FastAPI(
    title="ML Model",
    description="Description of the ML Model",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None,
    lifespan=lifespan
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder, like demo pages, if any
# app.mount("/static", StaticFiles(directory="static/"), name="static")

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

fake_db = {}

async def process_requests(q: asyncio.Queue, pool: ProcessPoolExecutor):
    while True:
        item = await q.get()
        loop = asyncio.get_running_loop()
        fake_db[item.id] = 'Processing'
        print("Processing")
        print(item)
        r = await loop.run_in_executor(pool, manage_prediction_request, item)
        q.task_done()
        fake_db[item.id] = 'Done'





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


    item_id = str(body.analysis_id)
    item = Item(id=item_id,
                ml_model_path=body.ml_model_path,
                file_path=body.file_path,
                analysis_id=body.analysis_id,
                parent_img_id=body.parent_img_id,
                ml_model_id=body.ml_model_id)
    
    background_tasks.add_task(request.state.q.put_nowait, item)
    fake_db[item_id] = 'Pending'

    return {"error": False}

    # try:
    #     background_tasks.add_task(manage_prediction_request, package)
    #     return {"error": False}
    # except Exception as e:
    #     print(e)
    #     return {"error": True}




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