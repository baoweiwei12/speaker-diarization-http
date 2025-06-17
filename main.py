from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, cast
from uuid import uuid4
import tempfile
import threading
import time
import json
import torch
import os
import redis
from pyannote.audio import Pipeline
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        if not self.HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable is required")
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_RESULT_EXPIRE = int(os.getenv("REDIS_RESULT_EXPIRE", "3600"))
        self.TASK_QUEUE_KEY = "diarization_tasks"
        self.TASK_RESULT_PREFIX = "diarization_result:"


class DiarizationSegment(BaseModel):
    start: float
    end: float
    speaker: str

class DiarizationResult(BaseModel):
    status: str
    start_time: int
    end_time: Optional[int] = None
    result: Optional[List[DiarizationSegment]] = None
    error: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str

class DiarizationModel:
    def __init__(self, hf_token: str):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=hf_token
        )
        self.pipeline.to(torch.device("cuda"))

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        diarization = self.pipeline(file_path)
        return [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]


class TaskManager:
    def __init__(self, config: Config):
        self.redis = redis.Redis(
            host=config.REDIS_HOST, 
            port=config.REDIS_PORT, 
            decode_responses=True
        )
        self.config = config

    def create_task(self, file_path: str) -> str:
        task_id = str(uuid4())
        self.redis.setex(
            self.config.TASK_RESULT_PREFIX + task_id,
            self.config.REDIS_RESULT_EXPIRE,
            json.dumps({
                "status": "pending",
                "start_time": int(time.time())
            })
        )
        
        task = {"task_id": task_id, "file_path": file_path}
        self.redis.rpush(self.config.TASK_QUEUE_KEY, json.dumps(task))
        return task_id

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        data = self.redis.get(self.config.TASK_RESULT_PREFIX + task_id)
        if data is None:
            return None
        return cast(Dict[str, Any], json.loads(data)) # type: ignore

    def update_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        self.redis.setex(
            self.config.TASK_RESULT_PREFIX + task_id,
            self.config.REDIS_RESULT_EXPIRE,
            json.dumps(result)
        )


class DiarizationService:
    def __init__(self, config: Config):
        self.config = config
        self.model = DiarizationModel(config.HF_TOKEN) # type: ignore
        self.task_manager = TaskManager(config)

    def process_file(self, file_path: str) -> str:
        return self.task_manager.create_task(file_path)

    def get_result(self, task_id: str) -> DiarizationResult:
        result = self.task_manager.get_task_result(task_id)
        if not result:
            return DiarizationResult(
                status="error",
                start_time=int(time.time()),
                error="Task not found or expired"
            )
        return DiarizationResult(**result)


class DiarizationAPI:
    def __init__(self):
        self.config = Config()
        self.service = DiarizationService(self.config)
        self.app = FastAPI(
            title="Speaker Diarization API",
            description="This is a Speaker Diarization API, which is used to diarize the audio file."
        )
        self._setup_routes()
        self._start_worker()

    def _setup_routes(self):
        @self.app.post("/diarize", response_model=TaskResponse)
        async def diarize(file: UploadFile = File(...)):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(await file.read())
            tmp.close()
            
            task_id = self.service.process_file(tmp.name)
            return TaskResponse(task_id=task_id)

        @self.app.get("/result/{task_id}", response_model=DiarizationResult)
        def get_result(task_id: str):
            return self.service.get_result(task_id)

    def _worker(self):
        while True:
            result = self.service.task_manager.redis.blpop([self.config.TASK_QUEUE_KEY])
            if result is None:
                continue
            _, task_data = result # type: ignore
            task = json.loads(task_data)
            task_id = task["task_id"]
            file_path = task["file_path"]
            start_time = int(time.time())
            
            try:
                result = self.service.model.process(file_path)
                self.service.task_manager.update_task_result(
                    task_id,
                    {
                        "status": "done",
                        "start_time": start_time,
                        "end_time": int(time.time()),
                        "result": result
                    }
                )
            except Exception as e:
                self.service.task_manager.update_task_result(
                    task_id,
                    {
                        "status": "error",
                        "start_time": start_time,
                        "end_time": int(time.time()),
                        "error": str(e)
                    }
                )

    def _start_worker(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def run(self, host: str = "0.0.0.0", port: int = 9001):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    api = DiarizationAPI()
    api.run()
