import os
import sys
import uuid
import shutil
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, os.path.abspath("src"))

from srt_translator.config import TranslatorConfig
from srt_translator.parser import parse_srt, save_srt
from srt_translator.merger import init_spacy_model, merge_entries_batch
from srt_translator.glossary import Glossary
from srt_translator.llm_client import create_client
from srt_translator.pipeline import TranslationPipeline

app = FastAPI(title="AISubtitleTranslator API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path("workspace")
WORK_DIR.mkdir(exist_ok=True)

JOB_STATE = {}
JOB_CLEANUP_INTERVAL = 300  # 每 5 分钟检查一次过期任务
JOB_RETENTION_SECONDS = 600  # 完成任务保留 10 分钟后清理
MAX_LOGS = 200  # 每个任务最大日志条数
MAX_CONCURRENT_JOBS = 5  # 最大并发翻译任务数

logger = logging.getLogger("api_server")


def log_msg(job_id: str, msg: str, is_error: bool = False):
    if job_id in JOB_STATE:
        prefix = "[错误] " if is_error else "- "
        logs = JOB_STATE[job_id]["logs"]
        logs.append({"text": prefix + msg, "isError": is_error})
        # 限制日志数量，保留最新的 MAX_LOGS 条
        if len(logs) > MAX_LOGS:
            # 保留开头和结尾的重要日志
            logs[:] = logs[:10] + logs[-(MAX_LOGS - 10):]


async def cleanup_expired_jobs():
    """定期清理过期的 JOB_STATE 条目。"""
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL)
        now = time.time()
        expired_ids = [
            job_id for job_id, state in JOB_STATE.items()
            if state.get("completed_at") is not None
            and now - state["completed_at"] > JOB_RETENTION_SECONDS
        ]
        for job_id in expired_ids:
            del JOB_STATE[job_id]
            logger.info(f"Cleaned up expired job: {job_id}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_expired_jobs())


async def process_translation_job(
    job_id: str,
    input_path: Path, 
    output_path: Path, 
    config: TranslatorConfig,
    summary_prompt: Optional[str] = None,
    translation_prompt: Optional[str] = None,
    glossary_text: str = ""
):
    try:
        JOB_STATE[job_id]["status"] = "running"
        JOB_STATE[job_id]["progress_pct"] = 2
        log_msg(job_id, "开始读取字幕文件...")
        
        content = input_path.read_text(encoding="utf-8-sig")
        entries = parse_srt(content)
        if not entries: 
            raise ValueError("字幕文件为空或格式不正确。")
            
        log_msg(job_id, f"读取成功，共包含 {len(entries)} 句字幕。")
        JOB_STATE[job_id]["progress_pct"] = 5
        
        if config.enable_merge:
            log_msg(job_id, "正在智能合并短句...")
            init_spacy_model()
            merged_entries = merge_entries_batch(entries, config.max_chars_per_entry, config.merge_time_gap)
            log_msg(job_id, f"句子合并完成，共计 {len(merged_entries)} 个翻译块。")
        else:
            merged_entries = [e.copy() for e in entries]
            
        client = create_client(config.api_key, config.base_url)
        
        # 构建 Glossary 对象（直接从字符串解析，避免文件并发竞争）
        glossary_obj = Glossary()
        if glossary_text and glossary_text.strip():
            import re
            for line in glossary_text.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # 支持 = 和 -> 两种分隔符
                match = re.match(r'^(.+?)\s*(?:=|->)\s*(.+)$', line)
                if match:
                    term, translation = match.groups()
                    glossary_obj.add(term, translation)
            if glossary_obj:
                log_msg(job_id, "已载入自定义术语表。")
        
        JOB_STATE[job_id]["progress_pct"] = 10
        log_msg(job_id, "正在翻译中...")
        
        # 进度回调：更新 JOB_STATE
        def _normalize_progress(job_id: str):
            """返回一个闭包，用于更新 JOB_STATE 进度。"""
            def callback(chunk_idx: int, pct: int):
                # pct: 0-100, 映射到 10-95 区间
                mapped_pct = 10 + int(pct * 0.85)
                JOB_STATE[job_id]["progress_pct"] = mapped_pct
            return callback
        
        # 创建管道并执行翻译
        # API 模式不传 progress_path，因为 API 任务是短生命周期的，
        # 不需要磁盘持久化进度文件来支持断点续翻
        pipeline = TranslationPipeline(config)
        translations = await pipeline.run(
            entries=merged_entries,
            glossary=glossary_obj,
            client=client,
            on_progress=_normalize_progress(job_id),
            summary_prompt=summary_prompt,
            translation_prompt=translation_prompt,
        )
        
        JOB_STATE[job_id]["progress_pct"] = 95
        log_msg(job_id, "翻译结束，正在保存文件...")
        
        final_entries = [entry.copy(text=translations.get(i, entry.text)) for i, entry in enumerate(merged_entries)]
        save_srt(final_entries, output_path)
        
        JOB_STATE[job_id]["progress_pct"] = 100
        JOB_STATE[job_id]["status"] = "completed"
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, f"全部任务完成！文件位置: {output_path}")
        
    except Exception as e:
        JOB_STATE[job_id]["status"] = "error"
        JOB_STATE[job_id]["error"] = str(e)
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, str(e), is_error=True)


@app.post("/api/translate")
async def translate_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(""),
    base_url: str = Form("https://api.deepseek.com"),
    summary_model_name: str = Form("deepseek-v4-pro"),
    model_name: str = Form("deepseek-v4-pro"),
    summary_prompt: str = Form(None),
    translation_prompt: str = Form(None),
    glossary: str = Form(""),
    save_path: str = Form(""),
    concurrency: int = Form(8)
):
    # 检查并发任务数
    running_jobs = sum(
        1 for s in JOB_STATE.values()
        if s.get("status") in ("running", "pending")
    )
    if running_jobs >= MAX_CONCURRENT_JOBS:
        return {
            "status": "error",
            "error": f"服务器繁忙，当前已有 {running_jobs} 个任务在运行（最大并发: {MAX_CONCURRENT_JOBS}）。请稍后再试。"
        }
    
    job_id = str(uuid.uuid4())
    input_path = WORK_DIR / f"{job_id}_{file.filename}"
    
    if save_path and save_path.strip():
        out_dir = Path(save_path.strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"translated_{file.filename}"
    else:
        output_path = Path.cwd() / f"translated_{file.filename}"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    config = TranslatorConfig(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        base_url=base_url, summary_model_name=summary_model_name,
        model_name=model_name, enable_merge=True, concurrency=concurrency
    )
    
    JOB_STATE[job_id] = {
        "status": "pending", "progress_pct": 0,
        "logs": [{"text": f"- 已接收文件: {file.filename}", "isError": False}],
        "error": None,
        "created_at": time.time(),
        "completed_at": None,
    }
    
    background_tasks.add_task(process_translation_job, job_id, input_path, output_path, config, summary_prompt, translation_prompt, glossary)
    
    return {"status": "success", "job_id": job_id, "expected_output": str(output_path)}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOB_STATE: return {"status": "error", "error": "任务 ID 不存在"}
    state = JOB_STATE[job_id]
    return {"status": state["status"], "progress": state["progress_pct"], "error": state["error"], "logs": state["logs"]}


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """取消一个正在执行或等待中的翻译任务。"""
    if job_id not in JOB_STATE:
        return {"status": "error", "error": "任务 ID 不存在"}
    
    state = JOB_STATE[job_id]
    current_status = state.get("status")
    
    if current_status in ("completed", "error"):
        return {"status": "error", "error": f"任务已结束（状态: {current_status}），无法取消。"}
    
    if current_status == "cancelled":
        return {"status": "error", "error": "任务已被取消。"}
    
    # 标记为取消状态
    JOB_STATE[job_id]["status"] = "cancelled"
    JOB_STATE[job_id]["completed_at"] = time.time()
    log_msg(job_id, "任务已被用户取消。")
    
    return {"status": "success", "job_id": job_id, "message": "任务已取消。"}


@app.get("/api/health")
async def health_check():
    """后端健康检查接口，供前端轮询判断服务是否就绪。"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
