#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from joblib import load
import redis
import json

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

redis_password = CONFIG['REDIS_PASSWORD'] if 'REDIS_PASSWORD' in CONFIG else None
if redis_password is not None:
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'redis'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        password=redis_password,
        )
else:

    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'redis'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        decode_responses=True
    )


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

    print('Running envirnoment: {}'.format(CONFIG['ENV']))
    
    q = asyncio.Queue()  # note that asyncio.Queue() is not thread safe
    pool = ProcessPoolExecutor(max_workers=1)

    # Check for pending tasks on startup
    try:
        print(f"Checking Redis for pending tasks...")
        pending_items = redis_client.keys("pending:*")
        processing_items = redis_client.keys("processing:*")
        all_items = processing_items + pending_items
        print("Found {0} pending/unfinished task(s): ".format(len(all_items)))
        print(all_items)
        for key in pending_items:
            item_data = json.loads(redis_client.get(key))
            # Create Item object from the stored data
            item = Item(
                id=item_data["id"],
                ml_model_path=item_data["ml_model_path"],
                file_path=item_data["file_path"],
                analysis_id=item_data["analysis_id"],
                parent_img_id=item_data["parent_img_id"],
                ml_model_id=item_data["ml_model_id"],
                dataset_id=item_data["dataset_id"],
                token=item_data["token"],
                debug=item_data["debug"],
                num_classes=item_data["num_classes"]
            )
            # Put the item in the queue
            await q.put(item)
            logger.info(f"Requeued pending analysis {item.id}")
    except Exception as e:
        logger.error(f"Error requeuing pending tasks: {e}")

    asyncio.create_task(process_requests(q, pool))  # Start the requests processing task
    yield {'q': q, 'pool': pool}
    pool.shutdown()


async def process_requests(q: asyncio.Queue, pool: ProcessPoolExecutor):
    while True:
        item = await q.get()
        loop = asyncio.get_running_loop()

        redis_client.set(f"processing:{item.id}", json.dumps(item.__dict__))
        redis_client.delete(f"pending:{item.id}")

        print("Processing")
        try:
            r = await loop.run_in_executor(pool, manage_prediction_request, item)
            print("Successful ??")
            print(r)

            redis_client.delete(f"processing:{item.id}")

        except Exception as e:
            print("Error")
            print(e)
            redis_client.set(f"failed:{item.id}", json.dumps({
                "item": item.__dict__,
                "error": str(e)
            }))
            redis_client.delete(f"processing:{item.id}")

            # update_analysis(analysis_id=item.analysis_id, token=item.token, completed=False, status="Error", error=str(e))
        q.task_done()



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

        redis_client.set(f"pending:{item.id}", json.dumps(item.__dict__))

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