from fastapi import APIRouter


def make_router(model_catalog: list[dict]) -> APIRouter:
    router = APIRouter()

    @router.get("/ai/health")
    def health():
        return {"status": "ok", "models": [m["id"] for m in model_catalog]}

    return router
