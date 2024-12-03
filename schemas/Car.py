from pydantic import BaseModel
from typing import List


class Car(BaseModel):
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    seats: int
    mark: str
    model: str
    year: int
    km_driven: int
    nm_torque: float
    bhp_max_power: float
    cc_engine: float
    kmpl_mileage: float


class Cars(BaseModel):
    objects: List[Car]
