from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Document QA API",
    version="0.1.0",
    description="Document question answering via an explicit RAG pipeline.",
)

app.include_router(router, prefix="/v1")


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        extra={"path": str(request.url), "error": str(exc)},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}
