from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from model_utils import predict_intent


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Banking Customer Intent Analysis",
    description="AI-powered banking customer intent classification",
    version="1.0.0"
)


# ==========================================
# Static Files
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==========================================
# Templates
# ==========================================

templates = Jinja2Templates(
    directory="templates"
)


# ==========================================
# Request Model
# ==========================================

class QueryRequest(BaseModel):
    text: str


# ==========================================
# Home Page
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ==========================================
# Prediction API
# ==========================================

@app.post("/predict")
async def predict(request: QueryRequest):

    text = request.text.strip()

    if not text:
        return {
            "error": "Please enter a banking query."
        }

    try:

        result = predict_intent(text)

        return result

    except Exception as e:

        return {
            "error": str(e)
        }


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
async def health():

    return {
        "status": "running",
        "project": "Banking Customer Intent Analysis"
    }
