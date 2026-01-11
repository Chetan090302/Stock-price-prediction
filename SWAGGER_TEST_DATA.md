# Test Data for Swagger API

Here are sample test data points you can use in the FastAPI Swagger UI to test the `/predict` endpoint.

## How to Access Swagger

1. Make sure the FastAPI server is running:
```bash
python app.py
```

2. Open your browser and go to:
```
http://localhost:8000/docs
```

## Test Data Examples

The model expects **normalized closing prices** (values between 0 and 1) from a sequence of 5 days.

### Example 1: Stable Prices
```json
{
  "data": [0.48, 0.50, 0.52, 0.51, 0.50]
}
```

### Example 2: Uptrend
```json
{
  "data": [0.40, 0.45, 0.50, 0.55, 0.60]
}
```

### Example 3: Downtrend
```json
{
  "data": [0.65, 0.60, 0.55, 0.50, 0.45]
}
```

### Example 4: High Volatility
```json
{
  "data": [0.45, 0.65, 0.40, 0.70, 0.35]
}
```

### Example 5: Real Data (from GOOG-year.csv normalized)
```json
{
  "data": [0.3780, 0.3760, 0.3840, 0.3920, 0.3900]
}
```

### Example 6: Recent Trend
```json
{
  "data": [0.50, 0.51, 0.52, 0.53, 0.54]
}
```

## Using Swagger UI

1. Navigate to `http://localhost:8000/docs`
2. Click on the **POST /predict** endpoint
3. Click the **"Try it out"** button
4. Replace the example JSON with one from above
5. Click **"Execute"** to see the prediction

## Expected Response

```json
{
  "prediction": 150.25,
  "scaled_prediction": 0.5023
}
```

- **prediction**: The stock price in original scale (dollars)
- **scaled_prediction**: The normalized prediction (0-1)

## Endpoints Available

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Make stock price prediction |
| `/docs` | GET | Interactive API documentation (Swagger) |
| `/redoc` | GET | ReDoc documentation |

## Tips

- The model expects a sequence of exactly 5 normalized prices
- Values should be between 0 and 1
- Try different patterns to see how the model responds
- Higher values in the sequence generally predict higher prices
- The prediction accuracy depends on the training data and model quality

## Running the API and Testing

```bash
# Terminal 1: Run the API
python app.py

# Terminal 2: Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [0.48, 0.50, 0.52, 0.51, 0.50]}'
```

## Normalized Data Explanation

The data in the CSV file uses **MinMax scaling**:
- Original Google stock price range: ~$700-$850
- Normalized range: 0 to 1
- Formula: (price - min) / (max - min)

So if you want to test with a specific price:
1. Get the min/max from training data
2. Apply the formula to normalize it
3. Use the normalized value in the API

Example conversions from GOOG stock prices:
- $750 → ~0.40 (normalized)
- $800 → ~0.50 (normalized)
- $850 → ~0.60 (normalized)
