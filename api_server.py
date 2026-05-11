import os
import sys
import uuid
import shutil
import time
import asyncio
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 基于 __file__ 的绝对路径，不依赖 cwd
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from srt_translator.config import TranslatorConfig
from srt_translator.parser import parse_srt, save_srt
from srt_translator.merger import (
    init_spacy_model_for_language_async,
    merge_entries_batch_async,
)
from srt_translator.glossary import load_glossary_from_string
from srt_translator.llm_client import create_client
from srt_translator.pipeline import TranslationPipeline
from srt_translator.quality_checker import run_quality_check


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动逻辑
    asyncio.create_task(cleanup_expired_jobs())
    asyncio.create_task(cleanup_workspace())
    yield  # 应用运行中
    # 关闭逻辑（可选）


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_server")

app = FastAPI(title="AISubtitleTranslator API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",        # Tauri 开发
        "http://127.0.0.1:1420",        # Tauri 开发（IP 访问）
        "http://tauri.localhost",       # Tauri 生产
        "tauri://localhost",            # Tauri 生产
        "https://tauri.localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _resolve_work_dir() -> Path:
    """Pick a writable runtime directory for uploaded temporary subtitles."""
    candidates = []
    if os.environ.get("AISUBTITLE_WORK_DIR"):
        candidates.append(Path(os.environ["AISUBTITLE_WORK_DIR"]))
    candidates.extend([
        Path(_current_dir) / ".runtime_workspace",
        Path(tempfile.gettempdir()) / "AISubtitleTranslator" / "workspace",
    ])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except Exception as exc:
            logger.warning("Runtime workspace unavailable %s: %s", candidate, exc)
    raise RuntimeError("No writable runtime workspace available")


WORK_DIR = _resolve_work_dir()

JOB_STATE: Dict[str, dict] = {}
JOB_TASKS: Dict[str, asyncio.Task] = {}  # 跟踪后台 asyncio 任务，用于真实取消
_concurrency_lock = asyncio.Lock()      # 并发控制锁
JOB_CLEANUP_INTERVAL = 300  # 每 5 分钟检查一次过期任务
JOB_RETENTION_SECONDS = 600  # 完成任务保留 10 分钟后清理
MAX_LOGS = 200  # 每个任务最大日志条数
MAX_CONCURRENT_JOBS = 5  # 最大并发翻译任务数


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
        try:
            await asyncio.sleep(JOB_CLEANUP_INTERVAL)
            now = time.time()
            expired_ids = [
                job_id for job_id, state in JOB_STATE.items()
                if state.get("completed_at") is not None
                and now - state["completed_at"] > JOB_RETENTION_SECONDS
            ]
            for job_id in expired_ids:
                try:
                    # 同时清理任务跟踪
                    JOB_TASKS.pop(job_id, None)
                    del JOB_STATE[job_id]
                    logger.info(f"Cleaned up expired job: {job_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up job {job_id}: {e}")
        except Exception as e:
            logger.error(f"Error in cleanup_expired_jobs: {e}")
            await asyncio.sleep(60)  # 避免快速重试


async def cleanup_workspace():
    """定期清理工作目录中超过 1 小时的临时文件。"""
    while True:
        await asyncio.sleep(3600)  # 每小时检查一次
        now = time.time()
        try:
            for f in WORK_DIR.iterdir():
                if f.is_file():
                    age = now - f.stat().st_mtime
                    if age > 3600:  # 超过 1 小时删除
                        f.unlink()
                        logger.info(f"Cleaned up temp file: {f.name}")
        except Exception:
            pass


async def process_translation_job(
    job_id: str,
    input_path: Path, 
    output_path: Path, 
    merged_output_path: Optional[Path],
    config: TranslatorConfig,
    summary_prompt: Optional[str] = None,
    translation_prompt: Optional[str] = None,
    glossary_text: str = "",
    merge_enabled: bool = True,
    save_merged_subtitles: bool = False,
    quality_check_enabled: bool = True,
):
    def _check_cancelled():
        """Raise CancelledError if this job has been cancelled via the API."""
        if JOB_STATE.get(job_id, {}).get("status") == "cancelled":
            raise asyncio.CancelledError()

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
        _check_cancelled()
        
        merge_completed = False
        if merge_enabled:
            log_msg(job_id, f"正在智能合并短句（spaCy NLP，源语言: {config.source_language}）...")
            try:
                await asyncio.wait_for(
                    init_spacy_model_for_language_async(config.source_language),
                    timeout=30.0,
                )
                _check_cancelled()
                merged_entries = await asyncio.wait_for(
                    merge_entries_batch_async(
                        entries,
                        config.max_chars_per_entry,
                        config.merge_time_gap,
                        source_language=config.source_language,
                    ),
                    timeout=60.0,
                )
                merge_completed = True
                log_msg(job_id, f"句子合并完成，共计 {len(merged_entries)} 个翻译块。")
            except asyncio.TimeoutError:
                log_msg(job_id, "智能合并超时，已跳过该步骤。", is_error=True)
                merged_entries = [e.copy() for e in entries]
            except asyncio.CancelledError:
                raise
            except Exception as merge_err:
                log_msg(job_id, f"智能合并失败: {merge_err}，已跳过该步骤。", is_error=True)
                merged_entries = [e.copy() for e in entries]
        else:
            log_msg(job_id, "已关闭字幕合并，将逐条翻译。")
            merged_entries = [e.copy() for e in entries]

        if save_merged_subtitles:
            if merge_enabled and merge_completed and merged_output_path is not None:
                save_srt(merged_entries, merged_output_path)
                log_msg(job_id, f"已保存 spaCy 合并后的字幕文件: {merged_output_path}")
            elif merge_enabled:
                log_msg(job_id, "spaCy 合并未成功，未保存合并字幕文件。", is_error=True)
            else:
                log_msg(job_id, "已关闭字幕合并，未保存合并字幕文件。")
        
        _check_cancelled()
        client = create_client(config.api_key, timeout=config.request_timeout)
        
        # 构建 Glossary 对象（使用 glossary 模块，避免重复解析逻辑）
        glossary_obj = load_glossary_from_string(glossary_text)
        if glossary_obj:
            log_msg(job_id, "已载入自定义术语表。")
        
        JOB_STATE[job_id]["progress_pct"] = 10
        log_msg(job_id, "正在翻译中...")
        
        def _normalize_progress(job_id: str):
            """返回一个异步闭包，用于更新 JOB_STATE 进度。"""
            async def callback(_chunk_idx: int, pct: int):
                # 翻译后如果还要质检，给质检阶段预留进度空间。
                upper_bound = 80 if quality_check_enabled else 95
                mapped_pct = 10 + int(pct * ((upper_bound - 10) / 100))
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

        translated_count = sum(
            1 for i, entry in enumerate(merged_entries)
            if translations.get(i, entry.text).strip() != entry.text.strip()
        )
        if merged_entries and translated_count == 0:
            raise RuntimeError(
                "没有成功翻译任何字幕。请检查 API Key、模型名称、账户余额或网络连接。"
            )
        
        final_entries = [entry.copy(text=translations.get(i, entry.text)) for i, entry in enumerate(merged_entries)]

        if quality_check_enabled:
            JOB_STATE[job_id]["progress_pct"] = 80
            log_msg(job_id, f"翻译结束，成功翻译 {translated_count}/{len(merged_entries)} 个块，正在自动质检译文...")

            async def _quality_progress(_chunk_idx: int, pct: int):
                JOB_STATE[job_id]["progress_pct"] = 80 + int(pct * 0.15)

            final_entries = await run_quality_check(
                original_entries=entries,
                translated_entries=final_entries,
                client=client,
                config=config,
                on_progress=_quality_progress,
            )
            _check_cancelled()
            log_msg(job_id, "自动质检完成，正在保存最终字幕文件...")
        else:
            JOB_STATE[job_id]["progress_pct"] = 95
            log_msg(job_id, f"翻译结束，成功翻译 {translated_count}/{len(merged_entries)} 个块，正在保存文件...")

        save_srt(final_entries, output_path)
        
        JOB_STATE[job_id]["progress_pct"] = 100
        JOB_STATE[job_id]["status"] = "completed"
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, f"全部任务完成！文件位置: {output_path}")
        
    except asyncio.CancelledError:
        # 任务被真正取消
        JOB_STATE[job_id]["status"] = "cancelled"
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, "任务已被取消。")
        raise  # 必须重新抛出以标记协程为已取消
    except Exception as e:
        JOB_STATE[job_id]["status"] = "error"
        JOB_STATE[job_id]["error"] = str(e)
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, str(e), is_error=True)
    finally:
        # 清理上传的临时文件
        try:
            if input_path.exists():
                input_path.unlink()
                logger.info(f"Cleaned up temp file: {input_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {input_path}: {e}")
        JOB_TASKS.pop(job_id, None)


@app.post("/api/translate")
async def translate_endpoint(
    file: UploadFile = File(...),
    api_key: str = Form(""),
    summary_model_name: str = Form("deepseek-v4-pro"),
    model_name: str = Form("deepseek-v4-pro"),
    summary_prompt: str = Form(None),
    translation_prompt: str = Form(None),
    glossary: str = Form(""),
    save_path: str = Form(""),
    concurrency: int = Form(8),
    max_output_tokens: int = Form(4096),
    request_timeout: float = Form(60.0),
    source_language: str = Form("en"),
    merge_enabled: bool = Form(True),
    save_merged_subtitles: bool = Form(False),
    quality_check_enabled: bool = Form(True),
):
    # 使用锁保护并发检查，避免竞态条件
    async with _concurrency_lock:
        running_jobs = sum(
            1 for s in JOB_STATE.values()
            if s.get("status") in ("running", "pending")
        )
        if running_jobs >= MAX_CONCURRENT_JOBS:
            return {
                "status": "error",
                "error": f"服务器繁忙，当前已有 {running_jobs} 个任务在运行（最大并发: {MAX_CONCURRENT_JOBS}）。请稍后再试。"
            }
    
    safe_name = Path(file.filename or "input.srt").name
    if not safe_name.lower().endswith(".srt"):
        return {"status": "error", "error": "仅支持 .srt 字幕文件"}

    job_id = str(uuid.uuid4())
    input_path = WORK_DIR / f"{job_id}_{safe_name}"
    
    if save_path and save_path.strip():
        # 去除用户可能粘贴的引号（如 "D:\Downloads" 或 'D:\Downloads'）
        raw_path = save_path.strip().strip("\"'")
        out_dir = Path(raw_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"status": "error", "error": f"保存路径不可用: {e}"}
        output_path = out_dir / f"translated_{safe_name}"
    else:
        # 跨平台获取桌面路径
        _home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME", "."))
        desktop = _home / "Desktop"
        if not desktop.exists():
            desktop = _home
        output_path = desktop / f"translated_{safe_name}"

    merged_output_path = output_path.with_name(f"merged_{safe_name}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    config = TranslatorConfig(
        api_key=api_key or None,
        summary_model_name=summary_model_name,
        model_name=model_name, concurrency=concurrency,
        max_output_tokens=max_output_tokens,
        request_timeout=request_timeout,
        source_language=source_language,
    )
    
    JOB_STATE[job_id] = {
        "status": "pending", "progress_pct": 0,
        "logs": [{"text": f"- 已接收文件: {safe_name}", "isError": False}],
        "error": None,
        "created_at": time.time(),
        "completed_at": None,
    }

    # 校验配置（如 api_key 为空时提前返回友好错误信息）
    error = config.validate()
    if error:
        JOB_STATE[job_id]["status"] = "error"
        JOB_STATE[job_id]["error"] = error
        JOB_STATE[job_id]["logs"].append({"text": f"[错误] {error}", "isError": True})
        JOB_STATE[job_id]["completed_at"] = time.time()
        # 清理已上传的临时文件
        if input_path.exists():
            input_path.unlink()
        return {"status": "error", "error": error, "job_id": job_id}

    # 使用 asyncio.create_task 创建可取消的任务，并保存引用
    task = asyncio.create_task(
        process_translation_job(
            job_id,
            input_path,
            output_path,
            merged_output_path,
            config,
            summary_prompt,
            translation_prompt,
            glossary,
            merge_enabled,
            save_merged_subtitles,
            quality_check_enabled,
        )
    )
    JOB_TASKS[job_id] = task
    
    response = {"status": "success", "job_id": job_id, "expected_output": str(output_path)}
    if save_merged_subtitles and merge_enabled:
        response["expected_merged_output"] = str(merged_output_path)
    return response


async def process_quality_check_job(
    job_id: str,
    original_path: Path,
    translated_path: Path,
    output_path: Path,
    config: TranslatorConfig,
):
    def _check_cancelled():
        if JOB_STATE.get(job_id, {}).get("status") == "cancelled":
            raise asyncio.CancelledError()

    try:
        JOB_STATE[job_id]["status"] = "running"
        JOB_STATE[job_id]["progress_pct"] = 2
        log_msg(job_id, "开始读取原字幕与译文字幕...")

        original_entries = parse_srt(original_path.read_text(encoding="utf-8-sig"))
        translated_entries = parse_srt(translated_path.read_text(encoding="utf-8-sig"))
        if not original_entries:
            raise ValueError("原字幕文件为空或格式不正确。")
        if not translated_entries:
            raise ValueError("译文字幕文件为空或格式不正确。")

        log_msg(job_id, f"读取成功：原字幕 {len(original_entries)} 条，译文字幕 {len(translated_entries)} 条。")
        JOB_STATE[job_id]["progress_pct"] = 10
        _check_cancelled()

        client = create_client(config.api_key, timeout=config.request_timeout)
        log_msg(job_id, "正在按时间轴对齐并检查译文质量...")

        async def _progress(_chunk_idx: int, pct: int):
            JOB_STATE[job_id]["progress_pct"] = 10 + int(pct * 0.85)

        corrected_entries = await run_quality_check(
            original_entries=original_entries,
            translated_entries=translated_entries,
            client=client,
            config=config,
            on_progress=_progress,
        )
        _check_cancelled()

        JOB_STATE[job_id]["progress_pct"] = 95
        log_msg(job_id, "质检完成，正在保存修正后的字幕文件...")
        save_srt(corrected_entries, output_path)

        JOB_STATE[job_id]["progress_pct"] = 100
        JOB_STATE[job_id]["status"] = "completed"
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, f"全部任务完成！文件位置: {output_path}")

    except asyncio.CancelledError:
        JOB_STATE[job_id]["status"] = "cancelled"
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, "任务已被取消。")
        raise
    except Exception as e:
        JOB_STATE[job_id]["status"] = "error"
        JOB_STATE[job_id]["error"] = str(e)
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, str(e), is_error=True)
    finally:
        for temp_path in (original_path, translated_path):
            try:
                if temp_path.exists():
                    temp_path.unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
        JOB_TASKS.pop(job_id, None)


@app.post("/api/quality-check")
async def quality_check_endpoint(
    original_file: UploadFile = File(...),
    translated_file: UploadFile = File(...),
    api_key: str = Form(""),
    model_name: str = Form("deepseek-v4-pro"),
    save_path: str = Form(""),
    concurrency: int = Form(8),
    max_output_tokens: int = Form(4096),
    request_timeout: float = Form(60.0),
):
    async with _concurrency_lock:
        running_jobs = sum(
            1 for s in JOB_STATE.values()
            if s.get("status") in ("running", "pending")
        )
        if running_jobs >= MAX_CONCURRENT_JOBS:
            return {
                "status": "error",
                "error": f"服务器繁忙，当前已有 {running_jobs} 个任务在运行（最大并发: {MAX_CONCURRENT_JOBS}）。请稍后再试。"
            }

    original_name = Path(original_file.filename or "original.srt").name
    translated_name = Path(translated_file.filename or "translated.srt").name
    if not original_name.lower().endswith(".srt") or not translated_name.lower().endswith(".srt"):
        return {"status": "error", "error": "仅支持 .srt 字幕文件"}

    job_id = str(uuid.uuid4())
    original_path = WORK_DIR / f"{job_id}_original_{original_name}"
    translated_path = WORK_DIR / f"{job_id}_translated_{translated_name}"

    if save_path and save_path.strip():
        raw_path = save_path.strip().strip("\"'")
        out_dir = Path(raw_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"status": "error", "error": f"保存路径不可用: {e}"}
        output_path = out_dir / f"corrected_{translated_name}"
    else:
        _home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME", "."))
        desktop = _home / "Desktop"
        if not desktop.exists():
            desktop = _home
        output_path = desktop / f"corrected_{translated_name}"

    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(original_file.file, buffer)
    with open(translated_path, "wb") as buffer:
        shutil.copyfileobj(translated_file.file, buffer)

    config = TranslatorConfig(
        api_key=api_key or None,
        summary_model_name=model_name,
        model_name=model_name,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
        request_timeout=request_timeout,
    )

    JOB_STATE[job_id] = {
        "status": "pending", "progress_pct": 0,
        "logs": [
            {"text": f"- 已接收原字幕: {original_name}", "isError": False},
            {"text": f"- 已接收译文字幕: {translated_name}", "isError": False},
        ],
        "error": None,
        "created_at": time.time(),
        "completed_at": None,
    }

    error = config.validate()
    if error:
        JOB_STATE[job_id]["status"] = "error"
        JOB_STATE[job_id]["error"] = error
        JOB_STATE[job_id]["logs"].append({"text": f"[错误] {error}", "isError": True})
        JOB_STATE[job_id]["completed_at"] = time.time()
        for temp_path in (original_path, translated_path):
            if temp_path.exists():
                temp_path.unlink()
        return {"status": "error", "error": error, "job_id": job_id}

    task = asyncio.create_task(
        process_quality_check_job(
            job_id,
            original_path,
            translated_path,
            output_path,
            config,
        )
    )
    JOB_TASKS[job_id] = task

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
    
    if current_status in ("completed", "error", "cancelled"):
        return {"status": "error", "error": f"任务已结束（状态: {current_status}），无法取消。"}
    
    # 立即标记为已取消，新的翻译请求会排除此任务
    # CancelledError 会异步传播到 process_translation_job 完成清理
    task = JOB_TASKS.get(job_id)
    if task and not task.done():
        JOB_STATE[job_id]["status"] = "cancelled"
        task.cancel()
        return {"status": "success", "job_id": job_id, "message": "取消信号已发送，正在停止任务..."}
    else:
        # 任务不存在或已完成，直接设为已取消
        JOB_STATE[job_id]["status"] = "cancelled"
        JOB_STATE[job_id]["completed_at"] = time.time()
        log_msg(job_id, "任务已被用户取消。")
        return {"status": "success", "job_id": job_id, "message": "任务已取消。"}


@app.get("/api/health")
async def health_check():
    """后端健康检查接口，供前端轮询判断服务是否就绪。"""
    return {"status": "ok"}


@app.get("/api/health/spacy")
async def spacy_health():
    """检查 spaCy 模型是否已加载。"""
    from srt_translator.merger import _nlp_model
    loaded = _nlp_model is not None
    return {
        "available": loaded,
        "message": "spaCy model loaded" if loaded else "spaCy model not initialized yet",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18770)
