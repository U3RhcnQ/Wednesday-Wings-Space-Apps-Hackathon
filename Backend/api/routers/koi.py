from fastapi import APIRouter

router = APIRouter(
    prefix="/koi",
    tags=["koi"],
)

@router.get("/")
async def read_koi_data():
    return {"message": "KOI data will be here"}

