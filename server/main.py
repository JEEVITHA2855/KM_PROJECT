from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ FIXED IMPORT (IMPORTANT)
from kmrl_unified_system import KMRLUnifiedMLSystem

# Create FastAPI app
app = FastAPI(
    title="KMRL Unified ML System",
    version="1.0.0"
)

# ================================
# ✅ LAZY LOAD ML MODEL (IMPORTANT)
# ================================
ml_system = None

def get_ml_system():
    global ml_system
    if ml_system is None:
        try:
            ml_system = KMRLUnifiedMLSystem()
            print("✅ ML system loaded successfully")
        except Exception as e:
            print(f"❌ ML load error: {e}")
            raise HTTPException(status_code=500, detail="ML system failed to load")
    return ml_system


# ================================
# REQUEST MODELS
# ================================
class AlertRequest(BaseModel):
    text: str

class EmbeddingRequest(BaseModel):
    text: str


# ================================
# HEALTH CHECK (IMPORTANT)
# ================================
@app.get("/")
def home():
    return {"status": "API running"}


# ================================
# ALERT ANALYSIS
# ================================
@app.post("/api/analyze")
async def analyze_alert(request: AlertRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    ml_system = get_ml_system()

    try:
        result = ml_system.classify_alert(request.text)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# EMBEDDING API
# ================================
@app.post("/api/embedding")
async def get_embedding(request: EmbeddingRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    ml_system = get_ml_system()

    try:
        embedding = ml_system.get_text_embedding(request.text)
        return {
            "text": request.text,
            "embedding": embedding.vector.tolist(),
            "dimension": embedding.dimension
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))