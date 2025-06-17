# 🎤 Speaker Diarization API

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

pyannote/speaker-diarization-3.1 的 HTTP API 封装

</div>

> 本项目基于 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 开发，提供了便捷的 HTTP API 接口。

## ✨ 功能特点

- 🎯 基于 pyannote/speaker-diarization-3.1 的高精度说话人分离
- 🔄 异步任务处理
- 💾 Redis 任务队列和结果存储
- 🐳 Docker 支持，一键部署

## 🚀 快速开始

### 方式一：Docker 部署（推荐）

```bash
docker pull baoweiwei2/speaker-diarization:latest
docker run -d \
    -p 9001:9001 \
    -e HF_TOKEN=your_huggingface_token \
    -e REDIS_HOST=your_redis_host \
    -e REDIS_RESULT_EXPIRE=3600 \
    baoweiwei2/speaker-diarization
```

### 方式二：本地部署

1. 克隆仓库
```bash
git clone https://github.com/baoweiwei12/speaker-diarization-http.git
cd speaker-diarization-http
```

2. 运行安装脚本
```bash
chmod +x install.sh
./install.sh
```

3. 配置环境变量
创建 `.env` 文件并设置以下变量：
```env
HF_TOKEN=your_huggingface_token
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_RESULT_EXPIRE=3600
```

4. 启动服务
```bash
python main.py
```

## 📝 API 使用说明

### API 文档

访问 `/docs` 路径可以查看完整的 Swagger API 文档：
```
http://localhost:9001/docs
```

### 1. 提交音频文件进行说话人分离

```bash
curl -X POST "http://localhost:9001/diarize" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio.wav"
```

响应示例：
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 2. 获取处理结果

```bash
curl -X GET "http://localhost:9001/result/{task_id}" \
     -H "accept: application/json"
```

响应示例：
```json
{
    "status": "done",
    "start_time": 1678901234,
    "end_time": 1678901235,
    "result": [
        {
            "start": 0.0,
            "end": 2.5,
            "speaker": "SPEAKER_00"
        },
        {
            "start": 2.5,
            "end": 5.0,
            "speaker": "SPEAKER_01"
        }
    ]
}
```

## 🔧 配置说明

在 `.env` 文件中可以修改以下配置：

- `HF_TOKEN`: HuggingFace API Token
- `REDIS_HOST`: Redis 服务器地址
- `REDIS_PORT`: Redis 服务器端口
- `REDIS_RESULT_EXPIRE`: 结果缓存过期时间（秒）

## 📄 开源协议

本项目采用 MIT 协议 - 查看 [LICENSE](LICENSE) 文件了解详情

---

