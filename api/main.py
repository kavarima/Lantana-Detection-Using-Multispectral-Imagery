import os
import sys
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Project layout:
# VEGETATION_INFERENCE/
#   api/main.py
#   inference_model.py
#   lantana_net_model_converted.pth
#
# The original inference_model.py is NOT modified.
PROJECT_DIR = Path(__file__).resolve().parent.parent
INFERENCE_SCRIPT = PROJECT_DIR / "inference_model.py"
MODEL_FILE = PROJECT_DIR / "lantana_net_model_converted.pth"

UPLOAD_ROOT = PROJECT_DIR / "uploads"
JOB_ROOT = PROJECT_DIR / "api_jobs"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
JOB_ROOT.mkdir(parents=True, exist_ok=True)

# The original script has a fixed INPUT_TIF filename.
EXPECTED_INPUT_FILENAME = "Site3_Oct_20261003_ms.tif"

UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024
MAX_UPLOAD_GB = float(os.getenv("MAX_UPLOAD_GB", "3"))
MAX_INFERENCE_WORKERS = 1
ALLOWED_EXTENSIONS = {".tif", ".tiff"}

app = FastAPI(
    title="Lantana Multispectral Inference API",
    description="Upload multispectral GeoTIFFs and download Lantana GeoJSON results.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend origin in production.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=MAX_INFERENCE_WORKERS)
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lantana-api")


def validate_project_files():
    missing = [
        str(path)
        for path in (INFERENCE_SCRIPT, MODEL_FILE)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError("Required project files are missing: " + ", ".join(missing))


def safe_filename(filename: str | None) -> str:
    return Path(filename or "uploaded.tif").name


def update_job(job_id: str, **changes):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(changes)


def update_image(job_id: str, image_id: str, **changes):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        for item in job["images"]:
            if item["image_id"] == image_id:
                item.update(changes)
                break
        job["completed_images"] = sum(
            x["status"] == "completed" for x in job["images"]
        )
        job["failed_images"] = sum(
            x["status"] == "failed" for x in job["images"]
        )


def public_job(job):
    result = {
        "job_id": job["job_id"],
        "status": job["status"],
        "total_images": job["total_images"],
        "completed_images": job["completed_images"],
        "failed_images": job["failed_images"],
        "created_at": job["created_at"],
        "images": [],
    }
    for image in job["images"]:
        item = {
            "image_id": image["image_id"],
            "filename": image["filename"],
            "status": image["status"],
        }
        if image.get("error"):
            item["error"] = image["error"]
        if image["status"] == "completed":
            item["geojson_url"] = (
                f"/api/jobs/{job['job_id']}/images/{image['image_id']}/geojson"
            )
            if image.get("summary_path"):
                item["summary_url"] = (
                    f"/api/jobs/{job['job_id']}/images/{image['image_id']}/summary"
                )
        result["images"].append(item)
    return result


def run_one_image(job_id: str, record: dict):
    image_id = record["image_id"]
    work_dir = JOB_ROOT / job_id / image_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy the original script and checkpoint byte-for-byte.
    script_copy = work_dir / "inference_model.py"
    model_copy = work_dir / "lantana_net_model_converted.pth"
    shutil.copy2(INFERENCE_SCRIPT, script_copy)
    shutil.copy2(MODEL_FILE, model_copy)

    # Save/copy the upload under the exact hard-coded name in the original script.
    expected_input = work_dir / EXPECTED_INPUT_FILENAME
    uploaded_path = Path(record["uploaded_path"])
    if uploaded_path.resolve() != expected_input.resolve():
        shutil.copy2(uploaded_path, expected_input)

    update_image(
        job_id, image_id, status="processing",
        started_at=datetime.now().astimezone().isoformat()
    )

    env = os.environ.copy()
    # The original script derives PROJECT_DIR from its own location.
    # It therefore reads the copied model/input and writes work_dir/inference.
    env.pop("INPUT_TIF", None)
    env.pop("OUTPUT_DIR", None)

    try:
        result = subprocess.run(
            [sys.executable, str(script_copy)],
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        output_dir = work_dir / "inference"
        geojson_path = output_dir / "lantana_polygons.geojson"
        summary_path = output_dir / "summary.json"

        if result.returncode != 0:
            raise RuntimeError(
                f"Inference exited with code {result.returncode}. "
                f"STDERR: {(result.stderr or '')[-10000:]}"
            )
        if not geojson_path.is_file():
            raise RuntimeError(f"GeoJSON was not created: {geojson_path}")

        update_image(
            job_id, image_id,
            status="completed",
            geojson_path=str(geojson_path),
            summary_path=str(summary_path) if summary_path.is_file() else None,
            log=(result.stdout or "")[-10000:],
            completed_at=datetime.now().astimezone().isoformat(),
        )
        logger.info("Completed job %s image %s", job_id, image_id)

    except Exception as exc:
        logger.exception("Inference failed: job=%s image=%s", job_id, image_id)
        update_image(job_id, image_id, status="failed", error=str(exc))


def process_job(job_id: str):
    update_job(job_id, status="processing")
    with jobs_lock:
        records = [dict(item) for item in jobs[job_id]["images"]]

    # Sequential processing: one large raster at a time.
    for record in records:
        run_one_image(job_id, record)

    with jobs_lock:
        job = jobs[job_id]
        completed = sum(x["status"] == "completed" for x in job["images"])
        failed = sum(x["status"] == "failed" for x in job["images"])
        job["completed_images"] = completed
        job["failed_images"] = failed
        job["status"] = (
            "completed" if failed == 0
            else "partial" if completed > 0
            else "failed"
        )


@app.get("/")
def root():
    return {
        "service": "Lantana Multispectral Inference API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "inference_script_exists": INFERENCE_SCRIPT.is_file(),
        "model_exists": MODEL_FILE.is_file(),
        "inference_workers": MAX_INFERENCE_WORKERS,
    }


@app.post("/api/inference", status_code=202)
async def submit_inference(
    images: List[UploadFile] = File(
        ...,
        description="Select one or more multispectral GeoTIFF files"
    )
):
    if not images:
        raise HTTPException(status_code=400, detail="Upload at least one TIFF.")

    try:
        validate_project_files()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    job_id = str(uuid.uuid4())
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        total_bytes = 0
        for upload in images:
            filename = safe_filename(upload.filename)
            extension = Path(filename).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file {filename!r}; upload .tif or .tiff."
                )

            image_id = str(uuid.uuid4())
            destination = upload_dir / f"{image_id}{extension}"
            file_bytes = 0

            # Bounded-memory upload: never read a multi-GB TIFF all at once.
            with destination.open("wb") as output:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)

                    if MAX_UPLOAD_GB > 0 and file_bytes > MAX_UPLOAD_GB * 1024**3:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"{filename} exceeds {MAX_UPLOAD_GB:g} GB. "
                                "Set MAX_UPLOAD_GB to a higher value."
                            )
                        )
                    output.write(chunk)

            await upload.close()
            if file_bytes == 0:
                raise HTTPException(status_code=400, detail=f"{filename} is empty.")

            records.append({
                "image_id": image_id,
                "filename": filename,
                "uploaded_path": str(destination),
                "size_bytes": file_bytes,
                "status": "queued",
            })

        free_bytes = shutil.disk_usage(PROJECT_DIR).free
        if free_bytes < 2 * 1024**3:
            raise HTTPException(
                status_code=507,
                detail="Less than 2 GB disk space remains; free space before inference."
            )

        job = {
            "job_id": job_id,
            "status": "queued",
            "total_images": len(records),
            "completed_images": 0,
            "failed_images": 0,
            "created_at": datetime.now().astimezone().isoformat(),
            "total_upload_bytes": total_bytes,
            "images": records,
        }
        with jobs_lock:
            jobs[job_id] = job

        executor.submit(process_job, job_id)
        return {
            "job_id": job_id,
            "status": "queued",
            "total_images": len(records),
            "total_upload_bytes": total_bytes,
            "message": "Upload complete; inference queued.",
            "status_url": f"/api/jobs/{job_id}",
        }

    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return public_job(job)


@app.get("/api/jobs/{job_id}/images/{image_id}/geojson")
def download_geojson(job_id: str, image_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        image = next(
            (x for x in job["images"] if x["image_id"] == image_id), None
        )
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found.")
        if image["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Image is not complete; status is {image['status']}."
            )
        path = Path(image["geojson_path"])

    if not path.is_file():
        raise HTTPException(status_code=404, detail="GeoJSON file not found.")
    return FileResponse(
        path=path,
        media_type="application/geo+json",
        filename=f"{image_id}_lantana_polygons.geojson",
    )


@app.get("/api/jobs/{job_id}/images/{image_id}/summary")
def download_summary(job_id: str, image_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        image = next(
            (x for x in job["images"] if x["image_id"] == image_id), None
        )
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found.")
        if image["status"] != "completed":
            raise HTTPException(status_code=409, detail="Inference is not complete.")
        summary_value = image.get("summary_path")

    if not summary_value:
        raise HTTPException(status_code=404, detail="Summary JSON was not generated.")
    path = Path(summary_value)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Summary file not found.")
    return FileResponse(
        path=path,
        media_type="application/json",
        filename=f"{image_id}_summary.json",
    )


@app.get("/api/jobs/{job_id}/images/{image_id}/files/{filename}")
def download_raster(job_id: str, image_id: str, filename: str):
    allowed = {"lantana_probability.tif", "lantana_mask.tif"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported output filename.")

    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        image = next(
            (x for x in job["images"] if x["image_id"] == image_id), None
        )
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found.")
        if image["status"] != "completed":
            raise HTTPException(status_code=409, detail="Inference is not complete.")
        geojson_path = Path(image["geojson_path"])

    path = geojson_path.parent / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Raster output not found.")
    return FileResponse(path=path, filename=filename, media_type="image/tiff")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
