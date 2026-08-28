from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    location: str
    carpet_area_sqft: float = Field(..., gt=0, description="Carpet area in square feet")
    floor_num: int = Field(..., ge=0, description="Floor number")
    bathroom: int = Field(..., ge=1, description="Number of bathrooms")
    balcony: int = Field(..., ge=0, description="Number of balconies")
    furnishing: str
    transaction: str
    ownership: str
    facing: str

class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "INR"
