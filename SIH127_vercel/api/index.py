import sys
import certifi
from fastapi import FastAPI, Request
from pymongo import MongoClient

app = FastAPI()

# Replace with your actual MongoDB connection string if needed
MONGO_URI = "mongodb+srv://user1:<db_password>@cluster0.rn7dha5.mongodb.net/?appName=Cluster0"

# MongoClient initialized once at module level to reuse connection pool across warm starts
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
)

db = client["traffic_system"]
logs_col = db["vehicle_logs"]


@app.get("/")
@app.get("/api")
def home():
    return {"status": "API is online"}


@app.post("/api/add-log")
async def add_vehicle_log(request: Request):
    try:
        log_data = await request.json()
    except Exception as parse_err:
        return {"status": "error", "message": f"Invalid JSON payload: {str(parse_err)}"}

    try:
        document = log_data.copy()

        # Format vehicle_id to uppercase if present
        if "vehicle_id" in document:
            document["vehicle_id"] = str(document["vehicle_id"]).upper()

        # Define unique constraint fields to prevent duplicate entries
        query_filter = {
            "cam_id": document.get("cam_id"),
            "vehicle_id": document.get("vehicle_id"),
            "time_stamp": document.get("time_stamp"),
        }

        # Update matching record or insert if it does not exist (Upsert)
        result = logs_col.update_one(
            query_filter,
            {"$set": document},
            upsert=True
        )

        return {
            "status": "success",
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None,
            "data": document,
        }
    except Exception as db_err:
        return {
            "status": "error",
            "error_type": type(db_err).__name__,
            "message": str(db_err),
            "python_version": sys.version,
        }
