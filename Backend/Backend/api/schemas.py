# This file will contain Pydantic models for data validation and serialization.
# These models define the structure of the data your API expects in requests
# and sends in responses.

from pydantic import BaseModel
from typing import List, Optional

class PlanetBase(BaseModel):
    # Define common base attributes for a planet
    name: str

class Planet(PlanetBase):
    # Attributes for a single planet response
    id: int
    orbital_period: Optional[float] = None

    class Config:
        orm_mode = True # This allows the model to be used with ORMs

class PlanetCreate(PlanetBase):
    # Attributes for creating a new planet
    pass

