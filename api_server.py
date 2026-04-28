import os
import sys
import uuid
import shutil
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

def log_msg(job_id: str, msg: str, is_error: bool = False):
    if job_id in JOB_STATE:
        prefix = "[错误] " if is_error else "- "
        JOB_STATE[job_id]["logs"].append({"text": prefix + msg, "isError": is_error})

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
        
        # --- 核心新增：动态覆写术语表文件 ---
        if glossary_text and glossary_text.strip():
            # 写入当前运行目录的 glossary.txt，供后端的 Glossary() 类读取
            with open("glossary.txt", "w", encoding="utf-8") as gf:
                gf.write(glossary_text.strip())
            log_msg(job_id, "已载入自定义术语表。")
        else:
            # 如果前端为空，清空本地术语表避免污染
            if os.path.exists("glossary.txt"):
                open("glossary.txt", "w", encoding="utf-8").close()
        
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
            
        all_texts = [e.text for e in merged_entries]
        client = create_client(config.api_key, config.base_url)
        
        # 构建 Glossary 对象（从 glossary.txt 文件读取）
        glossary_obj = Glossary()
        if os.path.exists("glossary.txt"):
            from srt_translator.glossary import load_glossary
            glossary_obj = load_glossary(Path("glossary.txt"))
        
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
        log_msg(job_id, f"全部任务完成！文件位置: {output_path}")
        
    except Exception as e:
        JOB_STATE[job_id]["status"] = "error"
        JOB_STATE[job_id]["error"] = str(e)
        log_msg(job_id, str(e), is_error=True)


@app.post("/api/translate")
async def translate_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(""),
    base_url: str = Form("https://api.deepseek.com"),
    summary_model_name: str = Form("deepseek-v4-pro"),
    model_name: str = Form("deepseek-v4-flash"),
    summary_prompt: str = Form(None),
    translation_prompt: str = Form(None),
    glossary: str = Form(""), # 新增：接收前端术语表
    save_path: str = Form(""),
    concurrency: int = Form(8)
):
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
        "error": None
    }
    
    background_tasks.add_task(process_translation_job, job_id, input_path, output_path, config, summary_prompt, translation_prompt, glossary)
    
    return {"status": "success", "job_id": job_id, "expected_output": str(output_path)}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOB_STATE: return {"status": "error", "error": "任务 ID 不存在"}
    state = JOB_STATE[job_id]
    return {"status": state["status"], "progress": state["progress_pct"], "error": state["error"], "logs": state["logs"]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)