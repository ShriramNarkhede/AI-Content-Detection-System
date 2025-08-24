from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

from core import AnalyzerCore

app = FastAPI(title="AI Content Detector API")
analyzer = AnalyzerCore()

# CORS (allow local dev frontends e.g., Vite on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    method: str = "Combined Analysis"


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    return analyzer.analyze_text(req.text, req.method)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


