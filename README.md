# Indian Food Image Classifier

Transfer-learning image classifier (MobileNetV3-Large) served through FastAPI with a Streamlit upload interface.

**Live app:** <STREAMLIT_URL>  
**Live API:** <RENDER_URL>/docs  (health: <RENDER_URL>/health)

## Problem statement
<Write 3-4 lines: who would use this, and what a wrong prediction costs them.>

## Dataset
- Source: <Kaggle/Hugging Face link>
- Classes used: <N> (min 150 images per class, capped at 300 for training speed)
- Split: 70 / 15 / 15 train / val / test, de-duplicated before splitting (see `artifacts/data_report.json`)

## Results (test set only)
| Model | Test accuracy |
|---|---|
| Baseline: frozen features + logistic regression | <xx.x%> |
| MobileNetV3-Large, head only (phase 1) | <optional> |
| MobileNetV3-Large, fine-tuned (phase 2) | <xx.x%> |

Training curves: `artifacts/curves.png` | Confusion matrix: `artifacts/confusion_matrix.png`

### Error analysis
<Summarise 10+ wrong predictions and why you think they happened: look-alike dishes, bad lighting, multiple dishes in one photo, etc.>

## Screenshots
<Add app screenshots, curves and confusion matrix here>

## Run it yourself
```bash
# 1. Train (GPU recommended, e.g. Google Colab)
pip install -r requirements-train.txt
python training/prepare_data.py      # needs data/raw/<class>/*.jpg
python training/train.py
python training/baseline.py
python training/evaluate.py
cp artifacts/model.onnx artifacts/classes.json api/model/

# 2. API
cd api && pip install -r requirements.txt
uvicorn main:app --reload            # http://localhost:8000/docs

# 3. Frontend
cd frontend && pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run app.py
```

## API
- `POST /predict` : multipart image upload, returns top-3 classes with confidence
- `GET /health` : service status and model version
- Validates file type (JPG/PNG/WEBP) and size (max 5 MB); model loads once at startup.

## Deployment
- API: Render (root directory `api`, build `pip install -r requirements.txt`, start `uvicorn main:app --host 0.0.0.0 --port $PORT`)
- Frontend: Streamlit Community Cloud with secret `API_URL`
