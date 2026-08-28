# House Price Prediction End-to-End ML Web Application

## 🏗️ Project Architecture & Structure

```text
house-price-project/
├── notebooks/
│   └── train_model.py          # Script for cleaning, preprocessing, model training & export
├── backend/
│   ├── app/
│   │   ├── schemas/
│   │   │   └── prediction.py   # Pydantic schema for request validation
│   │   ├── services/
│   │   │   └── preprocessing.py# Data cleaning & dataframe conversion
│   │   └── main.py             # FastAPI app with Lifespan context & prediction endpoints
│   ├── tests/
│   │   └── test_prediction.py  # Unit/Integration tests
│   ├── models/                 # Exported scikit-learn model pipeline (.pkl)
│   ├── locations.json          # Allowed location list exported from training data
│   ├── requirements.txt        # Python dependencies
│   └── .env.example            # Environment variables configuration
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── PredictionForm.tsx # Core React UI form component
│   │   ├── types/
│   │   │   └── prediction.ts   # TypeScript data models
│   │   ├── locations.json      # Dynamic location options for dropdowns
│   │   └── App.tsx             # Main React entry point
│   ├── .env.example            # Frontend environment variables
│   └── package.json            # Node.js project configuration
└── README.md
```

---

## 💡 System Architecture & Technical Explanation

### 1. Data Cleaning & Feature Engineering (`notebooks/`)
- **Price Standardization:** Text entries like `"42 Lac"` or `"1.2 Cr"` are parsed to floating-point numbers ($1 \text{ Lac} = 100,000$, $1 \text{ Cr} = 10,000,000$).
- **Area Normalization:** Extracted string units like `sqm` are converted directly to `sqft` ($1 \text{ sqm} \approx 10.764 \text{ sqft}$). Missing carpet area is imputed using super area values.
- **Categorical Binning:** High-cardinality categorical features such as `location` contain thousands of distinct listings. To keep model dimensions reasonable, only top-performing locations are tracked directly; all remaining sparse locations are assigned to `"other"`.

### 2. Bundled Machine Learning Pipeline
- **`ColumnTransformer` + `Pipeline`:** Instead of performing manual scaling or missing value imputation prior to serving, standard preprocessors (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) are chained directly within the exported `.pkl` object.
- **Benefits:** When raw data hits the API server, it directly passes into `model.predict()` without requiring separate step-by-step transformation scripts, preventing data leakage and reducing runtime bugs.

### 3. FastAPI Service (`backend/`)
- **Lifespan Context Manager:** Loads the model weights into memory **once** during startup. This avoids redundant disk reads per POST request, ensuring high throughput and fast response times.
- **Schema Validation:** `pydantic.BaseModel` validates input requests strictly (e.g., area must be non-zero), returning clear HTTP `422 Unprocessable Entity` responses when inputs fail constraints.

### 4. React Frontend (`frontend/`)
- **Dynamic Options:** Loads real location classes generated dynamically from the notebook training steps (`locations.json`).
- **Environment Configuration:** Reads base API URLs via `import.meta.env.VITE_API_BASE_URL` to easily handle differences between local development and production deployments.

---

## 🚀 How to Run

### 1. Train & Export the Model
```bash
Place your CSV at `notebooks/data/house_prices.csv`. The trainer accepts both `Amount(in rupees)` and `Amount (in rupees)`.

```bash
cd notebooks
python train_model.py
```

### 2. Run the FastAPI Backend
```bash
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Run the React Frontend
```bash
cd ../frontend
npm install
npm run dev
```
