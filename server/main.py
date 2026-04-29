from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from server.kmrl_unified_system import KMRLUnifiedMLSystem

# Create the FastAPI app
app = FastAPI(
    title="KMRL Unified ML System",
    description="An API for the KMRL Unified ML System for alert classification and analysis.",
    version="1.0.0"
)

# Check for the existence of the 'client/dist' directory
dist_dir = os.path.join(os.path.dirname(__file__), "client", "dist")
if os.path.exists(dist_dir):
    # Mount the static files directory (for the React frontend)
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")
else:
    print(f"Warning: Frontend build directory not found at {dist_dir}. The UI will not be available.")
    print("Please run 'npm --prefix client install' and 'npm --prefix client run build' to build the frontend.")

# Load the ML system
try:
    ml_system = KMRLUnifiedMLSystem()
    print("✅ KMRL Unified ML System loaded successfully.")
except Exception as e:
    print(f"❌ Critical error loading ML system: {e}")
    # In a real production scenario, you might want to handle this more gracefully
    # For now, we'll allow the app to start but log the error.
    ml_system = None

# Pydantic model for the request body
class AlertRequest(BaseModel):
    text: str

class EmbeddingRequest(BaseModel):
    text: str

# API endpoint for the root, serving the frontend's index.html
@app.get("/", include_in_schema=False)
async def read_index():
    index_path = os.path.join(dist_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found. Please build the client application.")

# API endpoint for alert classification
@app.post("/api/analyze")
async def analyze_alert(request: AlertRequest):
    """
    Analyzes an alert text and returns a classification.
    """
    if not ml_system:
        raise HTTPException(status_code=503, detail="ML system is not available.")
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        result = ml_system.classify_alert(request.text)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint for getting text embeddings
@app.post("/api/embedding")
async def get_embedding(request: EmbeddingRequest):
    """
    Generates and returns the embedding for a given text.
    """
    if not ml_system:
        raise HTTPException(status_code=503, detail="ML system is not available.")
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        embedding = ml_system.get_text_embedding(request.text)
        return {"text": request.text, "embedding": embedding.vector.tolist(), "dimension": embedding.dimension}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Catch-all for client-side routing
@app.get("/{catchall:path}", include_in_schema=False)
async def serve_react_app(catchall: str):
    index_path = os.path.join(dist_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="File not found.")

