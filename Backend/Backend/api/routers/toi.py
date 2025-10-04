from fastapi import APIRouter

router = APIRouter(
    prefix="/toi",
    tags=["toi"],
)

@router.get("/")
async def read_toi_data():
    return {"message": "TOI data will be here"}

