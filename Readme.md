# Lantana Multispectral Vegetation Inference

A Python-based system for detecting _Lantana camara_ from multispectral GeoTIFF imagery. The project uses a PyTorch neural network to estimate the probability of lantana presence, applies spatial post-processing, and exports the detected areas as GeoTIFF and GeoJSON outputs.

The repository also provides:

- A command-line/standalone inference workflow
- A FastAPI backend with asynchronous job handling
- A browser-based upload and download interface

## Features

- Multispectral inference using Green, Red, Red Edge, and NIR bands
- Band selection for a single stacked multispectral GeoTIFF
- Support for four separate aligned spectral rasters
- Resolution-aware removal of small and isolated detections
- Morphological closing, opening, and hole filling
- Probability, binary classification, GeoJSON, and JSON summary outputs
- CPU and CUDA GPU support
- Chunked raster processing for large images
- Upload API for one or more GeoTIFF files
- Simple web interface for running detections and downloading results

## Repository Structure

```text
Vegetation_Inference/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI application
├── inference_model.py           # Raster processing and model inference
├── lantana_net_model_converted.pth
├── check.py                     # Inspect TIFF bands and model checkpoint
├── index.html                   # Browser upload interface
├── Readme.md
└── .gitignore
```

Runtime files such as `inference/`, `uploads/`, `api_jobs/`, satellite imagery, and temporary files are ignored by Git.

## How It Works

1. Reads Green, Red, Red Edge, and NIR reflectance bands from the input raster.
2. Calculates:
   - NDVI
   - NDRE
   - GNDVI
   - NIR / Red Edge ratio
3. Normalizes the derived features using checkpoint-provided mean and standard deviation values.
4. Runs a small fully connected PyTorch network on CPU or CUDA.
5. Thresholds the predicted probability map.
6. Removes small objects and applies morphological cleaning.
7. Writes cleaned classification rasters and vector polygons.
8. Reprojects polygon geometry to WGS84 (EPSG:4326) for GeoJSON compatibility.

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- For local GPU inference: a CUDA-compatible NVIDIA GPU and matching PyTorch/CUDA installation

Main Python dependencies:

```text
numpy
rasterio
scipy
scikit-image
shapely
torch
fastapi
uvicorn
python-multipart
```

Install dependencies in a Windows PowerShell environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy rasterio scipy scikit-image shapely torch fastapi uvicorn python-multipart
```

## Input GeoTIFF Requirements

### Single stacked TIFF

The default standalone workflow expects a file named:

```text
Site3_Oct_20261003_ms.tif
```

The code expects the four bands at the following indices and descriptions:

| Band | Index | Description |
|---|---:|---|
| Green | 1 | Green |
| Red | 2 | Red |
| Red Edge | 3 | RedEdge |
| NIR | 4 | NIR |

The raster must:

- Contain the required bands
- Include a valid projected CRS
- Use metres as the CRS linear unit
- Contain georeferenced, finite reflectance values

Alpha is applied only when the GeoTIFF metadata identifies an alpha band. An extra fifth band is not assumed to be alpha.

### Separate spectral TIFFs

Set `INPUT_MODE = "separate"` in `inference_model.py` and configure the paths in `BAND_PATHS`:

```python
BAND_PATHS = {
    "green": PROJECT_DIR / "data" / "Green.tif",
    "red": PROJECT_DIR / "data" / "Red.tif",
    "rededge": PROJECT_DIR / "data" / "RedEdge.tif",
    "nir": PROJECT_DIR / "data" / "NIR.tif"
}
```

Each file must contain one spectral band. The implementation aligns different grids to the NIR raster grid using bilinear resampling.

## Standalone Inference

Place the input GeoTIFF in the project root and ensure `INPUT_MODE` is configured correctly.

Run:

```powershell
python inference_model.py
```

The script will use CUDA automatically if PyTorch detects a CUDA device; otherwise it will run on CPU.

## Inspect the Input and Model

Use `check.py` to inspect the stacked TIFF and checkpoint metadata:

```powershell
python check.py
```

This prints the TIFF band count, descriptions, color interpretations, and model parameter shapes.

## API Setup

Start the FastAPI server from the project root:

```powershell
python api/main.py
```

The service will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Alternatively, start it with Uvicorn:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Check API, model, and inference script availability |
| `POST` | `/api/inference` | Upload one or more `.tif`/`.tiff` images |
| `GET` | `/api/jobs/{job_id}` | Get job status and image results |
| `GET` | `/api/jobs/{job_id}/images/{image_id}/geojson` | Download detected polygons |
| `GET` | `/api/jobs/{job_id}/images/{image_id}/summary` | Download the inference summary |
| `GET` | `/api/jobs/{job_id}/images/{image_id}/files/{filename}` | Download a probability or classification TIFF |

Example upload:

```powershell
curl.exe -F "images=@Site3_Oct_20261003_ms.tif" http://127.0.0.1:8000/api/inference
```

The response includes a `job_id` and status URL. Poll the status URL until the job is `completed`, `partial`, or `failed`.

## Browser Interface

Open `index.html` directly in a browser after starting the API.

The interface:

- Accepts `.tif` or `.tiff` files by browsing or drag-and-drop
- Displays upload and inference status
- Downloads the generated GeoJSON
- Downloads the JSON summary when available

The API address is configured in `index.html`:

```javascript
const API = "http://127.0.0.1:8000";
```

Change this value when deploying the API to another host.

## Output Files

Standalone inference writes to `inference/`:

| File | Description |
|---|---|
| `lantana_probability.tif` | Continuous lantana probability raster |
| `lantana_mask.tif` | Binary classification raster: `1` lantana, `0` background, `255` invalid |
| `lantana_polygons.geojson` | WGS84 polygon features for detected lantana areas |
| `summary.json` | Resolution, threshold, pixel count, area, and output metadata |

GeoJSON polygon properties include:

- Polygon ID
- Class name
- Area in square metres
- Area in hectares
- Classification threshold

## Configuration

Important inference settings in `inference_model.py`:

| Setting | Purpose |
|---|---|
| `INPUT_MODE` | Select stacked or separate-band input |
| `ROW_CHUNK` | Number of raster rows processed at once |
| `GPU_BATCH` | Maximum inference batch size |
| `RESOLUTION_AWARE` | Scale cleaning parameters with raster resolution |
| `SIMPLIFY_METERS` | Simplification tolerance for exported polygons |
| `MIN_POLYGON_AREA_M2` | Minimum polygon area retained after export |

API settings in `api/main.py`:

| Environment variable | Default | Purpose |
|---|---:|---|
| `MAX_UPLOAD_GB` | `3` | Maximum size of an individual upload |

Job records and uploaded files are kept in memory and local directories. They are not persistent across an API process restart.

## Storage and Processing Limits

- One inference worker processes images sequentially
- Submission requires at least 2 GB of free disk space
- The API stores results under `api_jobs/`
- Uploaded files are retained under `uploads/`
- Large satellite rasters may require several minutes and substantial memory

## Model Checkpoint

The included `lantana_net_model_converted.pth` is loaded using PyTorch's `weights_only=True` mode with explicitly allowlisted NumPy metadata types. Load only a checkpoint obtained from a trusted source.

The checkpoint is expected to contain:

- `model_state`
- `mu` and `std` feature normalization values
- `eps`
- Optional `prob_threshold`

## Troubleshooting

### Missing CRS or unsupported CRS

The project requires a projected coordinate reference system whose linear unit is metres. Geographic CRS values such as EPSG:4326 are not accepted for standalone inference.

### Input raster has no CRS

Ensure the source file includes valid georeferencing metadata. For separate-band mode, all bands must expose a CRS and can be automatically aligned to the NIR grid.

### Band descriptions do not match

For stacked TIFF input, set band descriptions to `Green`, `Red`, `RedEdge`, and `NIR`. Bands must appear in the configured indices.

### CUDA is not used

If `Device: cpu` is printed, PyTorch is not detecting a CUDA environment. CPU inference remains supported.

### API cannot be reached

Confirm the server is running on port `8000` and that `const API` in `index.html` points to the correct address.

## Deployment Considerations

The current implementation is suitable for local use, demonstrations, and controlled environments. Before internet deployment:

- Restrict CORS to trusted frontend origins
- Add authentication and authorization
- Use a persistent job database and object storage
- Add request size and concurrent-job limits
- Add HTTPS and reverse-proxy configuration
- Introduce cleanup policies for uploads and temporary job directories
- Add automated tests and monitoring

## Disclaimer

Detected polygons are model predictions and should be validated against ground observations and appropriate field surveys before operational or ecological management decisions.
