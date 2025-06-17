from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
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

# 加载 .env 配置
load_dotenv()

# 读取配置
HF_TOKEN = os.getenv("HF_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_RESULT_EXPIRE = int(os.getenv("REDIS_RESULT_EXPIRE", "3600"))

# 初始化 FastAPI & Redis
app = FastAPI(title="Speaker Diarization API", description="This is a Speaker Diarization API, which is used to diarize the audio file.")
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# 初始化模型
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
pipeline.to(torch.device("cuda"))

TASK_QUEUE_KEY = "diarization_tasks"
TASK_RESULT_PREFIX = "diarization_result:"


# --------------------
# Pydantic 模型
# --------------------
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


# --------------------
# 后台 worker 线程
# --------------------
def worker():
    while True:
        _, task_data = r.blpop(TASK_QUEUE_KEY)  # type: ignore
        task = json.loads(task_data)
        task_id = task["task_id"]
        file_path = task["file_path"]
        start_time = int(time.time())
        try:
            diarization = pipeline(file_path)
            result = [
                {"start": turn.start, "end": turn.end, "speaker": speaker}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            r.setex(TASK_RESULT_PREFIX + task_id, REDIS_RESULT_EXPIRE, json.dumps({
                "status": "done",
                "start_time": start_time,
                "end_time": int(time.time()),
                "result": result
            }))
        except Exception as e:
            r.setex(TASK_RESULT_PREFIX + task_id, REDIS_RESULT_EXPIRE, json.dumps({
                "status": "error",
                "start_time": start_time,
                "end_time": int(time.time()),
                "error": str(e)
            }))

# 启动后台线程
threading.Thread(target=worker, daemon=True).start()


# --------------------
# 接口定义
# --------------------
@app.post("/diarize", response_model=TaskResponse)
async def diarize(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await file.read())
    tmp.close()

    task_id = str(uuid4())
    r.setex(TASK_RESULT_PREFIX + task_id, REDIS_RESULT_EXPIRE, json.dumps({
        "status": "pending",
        "start_time": int(time.time())
    }))

    task = {"task_id": task_id, "file_path": tmp.name}
    r.rpush(TASK_QUEUE_KEY, json.dumps(task))

    return {"task_id": task_id}


@app.get("/result/{task_id}", response_model=DiarizationResult)
def get_result(task_id: str):
    data= r.get(TASK_RESULT_PREFIX + task_id)
    if not data:
        return {
            "status": "error",
            "start_time": int(time.time()),
            "error": "Task not found or expired"
        }

    return json.loads(data) # type: ignore


# --------------------
# 本地运行入口
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
