from fastapi import FastAPI
from app.routes import analyze
from app.core.database import Base, engine
from app.routes import history
from fastapi.middleware.cors import CORSMiddleware

# CREATE TABLES (runs once on startup)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Skin Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(title="AI Skin Analyzer API")

app.include_router(analyze.router, prefix="/api")

app.include_router(history.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "API is running"}