# API Documentation

## Endpoints

### GET `/api/v1/data`
- Returns the latest climate data as JSON.
- **Query parameters:** `province`, `district`, `date_range`

### POST `/api/v1/predict`
- Returns model predictions for specific input features.
- **Payload:** JSON with feature values

### GET `/api/v1/status`
- Returns system and data pipeline status.

---

## Authentication

- All endpoints require an API key.
- Pass the API key in the `Authorization` header:  
  `Authorization: Bearer <your-api-key>`

## Examples

```bash
curl -H "Authorization: Bearer <api-key>" https://yourapp.com/api/v1/data?province=Bagmati
```

## Error Codes

- `401 Unauthorized`: Invalid or missing API key
- `400 Bad Request`: Missing/invalid parameters
- `500 Internal Server Error`: Server-side error

---

## Changelog

- v1.0: Initial API release