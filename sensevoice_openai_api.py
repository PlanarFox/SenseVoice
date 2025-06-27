#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
SenseVoice OpenAI-Compatible API
Provides OpenAI API compatible endpoints for SenseVoice STT
"""

import os
import io
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import numpy as np
import librosa
import soundfile as sf

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
app = FastAPI(title="SenseVoice OpenAI-Compatible API", version="1.0.0")

# Global model instance
model = None
actual_model = None

# Configuration from environment
MODEL_DIR = os.getenv("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")
DEVICE = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
API_KEY = os.getenv("SENSEVOICE_API_KEY", "")  # Empty means any key is accepted
VAD_MODEL = os.getenv("SENSEVOICE_VAD_MODEL", "fsmn-vad")
MAX_SINGLE_SEGMENT_TIME = int(os.getenv("SENSEVOICE_MAX_SINGLE_SEGMENT_TIME", "30000"))


class TranscriptionRequest(BaseModel):
    """OpenAI-compatible transcription request model"""
    model: str = Field(..., description="Model ID")
    language: Optional[str] = Field(None, description="Language code in ISO-639-1 format")
    prompt: Optional[str] = Field(None, description="Optional prompt (ignored)")
    response_format: Optional[str] = Field("json", description="Response format")
    temperature: Optional[float] = Field(0, description="Sampling temperature (ignored)")
    timestamp_granularities: Optional[List[str]] = Field(None, description="Timestamp options (ignored)")


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response"""
    text: str


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response"""
    error: Dict[str, Any]


def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Missing Authorization header", "type": "invalid_request_error"}}
        )
    
    # Extract token from "Bearer TOKEN" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid Authorization header format", "type": "invalid_request_error"}}
        )
    
    # Verify API key if configured
    if API_KEY and token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "invalid_api_key"}}
        )
    
    return token


def load_model():
    """Initialize the SenseVoice model"""
    global actual_model
    if actual_model is None:
        logger.info(f"Loading SenseVoice model from {MODEL_DIR}")
        actual_model = AutoModel(
            model=MODEL_DIR,
            trust_remote_code=True,
            vad_model=VAD_MODEL,
            vad_kwargs={"max_single_segment_time": MAX_SINGLE_SEGMENT_TIME},
            device=DEVICE,
            disable_update=True,
        )
        logger.info("Model loaded successfully")


def map_language_code(language: Optional[str]) -> str:
    """Map ISO-639-1 language codes to SenseVoice format"""
    if not language:
        return "zh"  # Default to Chinese
    
    # Direct mappings
    language_map = {
        "zh": "zh",      # Chinese
        "en": "en",      # English
        "ja": "ja",      # Japanese
        "ko": "ko",      # Korean
        "yue": "yue",    # Cantonese
        "zh-cn": "zh",   # Chinese (China)
        "zh-tw": "zh",   # Chinese (Taiwan)
        "zh-hk": "yue",  # Chinese (Hong Kong) -> Cantonese
    }
    
    # Return mapped language or "auto" for unsupported
    return language_map.get(language.lower(), "auto")


def convert_audio_to_wav(audio_bytes: bytes, filename: str) -> bytes:
    """Convert audio to WAV format if needed"""
    try:
        # Try to load with librosa (handles most formats)
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        
        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sr, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        return wav_buffer.read()
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Error processing audio file: {str(e)}", "type": "invalid_audio_format"}}
        )


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SenseVoice OpenAI-Compatible API", "version": "1.0.0"}


@app.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models (OpenAI-compatible)"""
    return {
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "sensevoice"
            },
            {
                "id": "gpt-4o-transcribe",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "sensevoice"
            },
            {
                "id": "gpt-4o-mini-transcribe",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "sensevoice"
            }
        ],
        "object": "list"
    }


@app.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    api_key: str = Depends(verify_api_key)
):
    """
    Create transcription (OpenAI-compatible)
    
    This endpoint transcribes audio files using SenseVoice model.
    All model parameters are mapped to the single SenseVoice model.
    """

    global actual_model
    
    # Validate response format
    if response_format != "json":
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Only 'json' response format is supported", "type": "invalid_request_error"}}
        )
    
    # Log request info
    logger.info(f"Transcription request: model={model}, language={language}, file={file.filename}")
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Convert to WAV if needed (SenseVoice handles various formats, but WAV is most reliable)
        if not file.filename.lower().endswith('.wav'):
            audio_bytes = convert_audio_to_wav(audio_bytes, file.filename)
        
        # Save to temporary file (SenseVoice needs file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Map language code
            sensevoice_language = map_language_code(language)
            logger.info(f"Using language: {sensevoice_language}")
            
            # Perform transcription
            result = actual_model.generate(
                input=tmp_path,
                cache={},
                language=sensevoice_language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            # Process result
            if result and len(result) > 0:
                text = rich_transcription_postprocess(result[0]["text"])
                logger.info(f"Transcription successful: {len(text)} characters")
                
                # Return OpenAI-compatible response
                return TranscriptionResponse(text=text)
            else:
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"message": "No transcription result", "type": "transcription_error"}}
                )
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Transcription failed: {str(e)}", "type": "transcription_error"}}
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions in OpenAI format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"error": {"message": exc.detail, "type": "error"}}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions in OpenAI format"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "internal_error"}}
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("SENSEVOICE_API_HOST", "0.0.0.0")
    port = int(os.getenv("SENSEVOICE_API_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
