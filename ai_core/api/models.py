from fastapi import APIRouter


def make_router(model_catalog: list[dict]) -> APIRouter:
    router = APIRouter()

    @router.get("/ai/models")
    def list_models() -> list[dict]:
        return model_catalog

    return router
