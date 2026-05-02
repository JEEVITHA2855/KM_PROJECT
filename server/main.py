from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ml_system = None

def get_ml_system():
    global ml_system
    if ml_system is None:
        try:
            print("🔄 Loading ML model...")
            from kmrl_unified_system import KMRLUnifiedMLSystem
            ml_system = KMRLUnifiedMLSystem()
            print("✅ Model loaded")
        except Exception as e:
            print("❌ ERROR LOADING MODEL:", e)
            raise HTTPException(status_code=500, detail=str(e))
    return ml_system


class AlertRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/api/analyze")
def analyze(req: AlertRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Empty input")

    ml = get_ml_system()

    try:
        result = ml.classify_alert(req.text)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)