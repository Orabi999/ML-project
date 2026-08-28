import pandas as pd
from typing import List, Dict, Any

def build_input_dataframe(request_data: Any, allowed_locations: List[str]) -> pd.DataFrame:
    loc = request_data.location if request_data.location in allowed_locations else "other"
    
    data = {
        "carpet_area_sqft": [request_data.carpet_area_sqft],
        "floor_num": [request_data.floor_num],
        "Bathroom": [request_data.bathroom],
        "Balcony": [request_data.balcony],
        "location_grouped": [loc],
        "Furnishing": [request_data.furnishing],
        "Transaction": [request_data.transaction],
        "Ownership": [request_data.ownership],
        "facing": [request_data.facing],
    }
    return pd.DataFrame(data)
