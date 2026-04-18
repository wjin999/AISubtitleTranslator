import os
import sys
import asyncio
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
from srt_translator.translator import generate_context_summary, translate_chunk_task
from srt_translator.progress import TranslationProgress
from srt_translator.text_utils import clean_translated_text

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
        
        chunk_size = config.chunk_size
        chunks = [all_texts[i:i + chunk_size] for i in range(0, len(all_texts), chunk_size)]
        
        progress_path = WORK_DIR / f"{job_id}.progress.json"
        progress = TranslationProgress.create(str(input_path), len(chunks))
        
        JOB_STATE[job_id]["progress_pct"] = 10
        log_msg(job_id, "正在生成全文摘要...")
        
        summary = await generate_context_summary("\n".join(all_texts), client, config.summary_model_name)
        
        JOB_STATE[job_id]["progress_pct"] = 15
        log_msg(job_id, "摘要生成完毕，开始批量翻译...")
        
        sem = asyncio.Semaphore(config.concurrency)
        results_dict = {}
        
        async def _do_chunk(chunk_idx):
            chunk = chunks[chunk_idx]
            start_idx = chunk_idx * config.chunk_size
            end_idx = start_idx + len(chunk)
            
            prev_start = max(0, start_idx - config.context_window)
            context_prev = all_texts[prev_start:start_idx]
            next_end = min(len(all_texts), end_idx + config.context_window)
            context_next = all_texts[end_idx:next_end]
            
            chunk_data = [{"index": start_idx + j, "text": text} for j, text in enumerate(chunk)]
            
            res = await translate_chunk_task(
                client=client, chunk_data=chunk_data, context_prev=context_prev,
                context_next=context_next, global_summary=summary, glossary=Glossary(),
                model=config.model_name, sem=sem
            )
            return chunk_idx, res

        tasks = [_do_chunk(i) for i in range(len(chunks))]
        total_chunks = len(tasks)
        completed_count = 0
        
        for coro in asyncio.as_completed(tasks):
            chunk_idx, chunk_results = await coro
            chunk_translations = {}
            for r in chunk_results:
                if r.success:
                    cleaned = clean_translated_text(r.translated)
                    results_dict[r.index] = cleaned
                    chunk_translations[r.index] = cleaned
                else:
                    results_dict[r.index] = r.original
                    chunk_translations[r.index] = r.original
            
            if progress:
                progress.mark_completed(chunk_idx, chunk_translations)
            
            completed_count += 1
            JOB_STATE[job_id]["progress_pct"] = 15 + int((completed_count / total_chunks) * 80)
            
        JOB_STATE[job_id]["progress_pct"] = 95
        log_msg(job_id, "翻译结束，正在保存文件...")
        
        final_entries = [entry.copy(text=results_dict.get(i, entry.text)) for i, entry in enumerate(merged_entries)]
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
    summary_model_name: str = Form("deepseek-reasoner"),
    model_name: str = Form("deepseek-chat"),
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