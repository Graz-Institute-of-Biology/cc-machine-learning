#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, Json
import numpy as np


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    file_path: str = Field(..., example='12/images/img_1.jpg', title='Path to the input image file')
    model_path: str = Field(..., example='12/models/model_1.pt', title='Path to the model file')
    analysis_id: int = Field(..., example=1, title='Analysis ID')

class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    mask: Json = Field(..., example='12/masks/mask_1.jpg', title='Path to the output mask file')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')