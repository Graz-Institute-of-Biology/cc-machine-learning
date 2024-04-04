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
from ml_api import manage_prediction_request, update_analysis
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from concurrent.futures import ProcessPoolExecutor
import asyncio
from dataclasses import dataclass
from fastapi.middleware.cors import CORSMiddleware



@dataclass
class Item:
    id: str
    ml_model_path: str
    file_path: str
    analysis_id: int
    parent_img_id: int
    ml_model_id: int
    dataset_id: int
    token: str
    debug: bool
    num_classes: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info('Running envirnoment prody: {}'.format(CONFIG['ENV']))
    # logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))
    
    q = asyncio.Queue()  # note that asyncio.Queue() is not thread safe
    pool = ProcessPoolExecutor(max_workers=1)
    asyncio.create_task(process_requests(q, pool))  # Start the requests processing task
    yield {'q': q, 'pool': pool}
    pool.shutdown()

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


if CONFIG['ENV'] == 'development':
    origins = [ CONFIG['HOST'] ]
    # print(origins)
    app.add_middleware(CORSMiddleware, allow_origins=origins)
elif CONFIG['ENV'] == 'staging':
    origins = [ CONFIG['HOST'] ]
    app.add_middleware(CORSMiddleware, allow_origins=origins)
elif CONFIG['ENV'] == 'production':
    origins = [ CONFIG['HOST'] ]
    app.add_middleware(CORSMiddleware, allow_origins=origins)

# Mount static folder, like demo pages, if any
# app.mount("/static", StaticFiles(directory="static/"), name="static")

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

async def process_requests(q: asyncio.Queue, pool: ProcessPoolExecutor):
    while True:
        item = await q.get()
        loop = asyncio.get_running_loop()
        print("Processing")
        try:
            r = await loop.run_in_executor(pool, manage_prediction_request, item)
            print("Successful ??")
            print(r)
        except Exception as e:
            print("Error")
            print(e)
            # update_analysis(analysis_id=item.analysis_id, token=item.token, completed=False, status="Error", error=str(e))
        q.task_done()





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
    logger.info('Handling request: {}'.format(body))
    print("Handling request: {}".format(body))

    try:
        item_id = str(body.analysis_id)
        item = Item(id=item_id,
                    ml_model_path=body.ml_model_path,
                    file_path=body.file_path,
                    analysis_id=body.analysis_id,
                    parent_img_id=body.parent_img_id,
                    ml_model_id=body.ml_model_id,
                    debug=body.debug,
                    dataset_id=body.dataset_id,
                    token=body.token,
                    num_classes=body.num_classes
                    )
        update_analysis(analysis_id=body.analysis_id, token=body.token, completed=False, status="Received/Queued")
        background_tasks.add_task(request.state.q.put_nowait, item)

        return {"error": False}
    except Exception as e:
        logger.info('ERROR: {}'.format(e))
        error = traceback.format_exc()
        print("ERROR:")
        print(error)
        update_analysis(analysis_id=body.analysis_id, token=body.token, completed=False, status="Error", error=error)
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