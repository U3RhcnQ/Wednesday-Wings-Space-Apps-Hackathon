from fastapi import APIRouter

router = APIRouter(
    prefix="/k2",
    tags=["k2"],
)

@router.get("/")
async def read_k2_data():
    return {"message": "K2 data will be here"}

