import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import timedelta

app = FastAPI(title="Tech Resale Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLOUD-READY DATABASE CONNECTION ---
# This tells the app: "If we are on Render, use the cloud URL. If we are on your computer, use localhost."
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "postgresql://admin:password123@localhost:5432/resale_predictor"
)

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
# ---------------------------------------

@app.on_event("startup")
def startup_db_client():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fresh reset: Drop old tables to clear out old setups
        cursor.execute("DROP TABLE IF EXISTS historical_prices CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS devices CASCADE;")
        
        # Recreate tables cleanly
        cursor.execute("""
            CREATE TABLE devices (
                device_id SERIAL PRIMARY KEY,
                brand VARCHAR(50),
                model VARCHAR(100),
                original_msrp NUMERIC,
                release_date DATE
            );
            CREATE TABLE historical_prices (
                record_id SERIAL PRIMARY KEY,
                device_id INTEGER REFERENCES devices(device_id),
                date_recorded DATE,
                condition VARCHAR(50),
                resale_price NUMERIC,
                data_source VARCHAR(100)
            );
        """)
        
        print("Populating database with 10 devices and historical data...")
        
        # Insert 10 distinct devices
        devices_data = [
            ('Samsung', 'Galaxy Tab S8 Ultra', 40000, '2022-02-09'),
            ('DJI', 'Osmo Action 4', 14500, '2023-08-02'),
            ('GoPro', 'Hero 12 Black', 15000, '2023-09-13'),
            ('Apple', 'iPhone 15 Pro', 41900, '2023-09-22'),
            ('Samsung', 'Galaxy S24 Ultra', 46900, '2024-01-17'),
            ('Apple', 'MacBook Air M3', 39900, '2024-03-08'),
            ('Sony', 'WH-1000XM5', 14900, '2022-05-20'),
            ('DJI', 'Mini 4 Pro', 34500, '2023-09-25'),
            ('Nintendo', 'Switch OLED', 12900, '2021-10-08'),
            ('Apple', 'iPad Pro M4', 39900, '2024-05-07')
        ]
        
        for dev in devices_data:
            cursor.execute(
                "INSERT INTO devices (brand, model, original_msrp, release_date) VALUES (%s, %s, %s, %s) RETURNING device_id;",
                dev
            )
            dev_id = cursor.fetchone()['device_id']
            
            # Generate 10 downward trending historical price data points per device
            msrp = dev[2]
            prices = [
                int(msrp * 0.85), int(msrp * 0.80), int(msrp * 0.76), int(msrp * 0.72), int(msrp * 0.68),
                int(msrp * 0.65), int(msrp * 0.61), int(msrp * 0.58), int(msrp * 0.55), int(msrp * 0.52)
            ]
            dates = [
                '2024-01-15', '2024-04-15', '2024-07-15', '2024-10-15', '2025-01-15',
                '2025-04-15', '2025-07-15', '2025-10-15', '2026-01-15', '2026-04-15'
            ]
            
            for d, p in zip(dates, prices):
                cursor.execute(
                    "INSERT INTO historical_prices (device_id, date_recorded, resale_price, condition, data_source) VALUES (%s, %s, %s, 'Good', 'Marketplace');",
                    (dev_id, d, p)
                )
                
        conn.commit()
        print("Database successfully loaded with 10 devices!")
    except Exception as e:
        print(f"Startup DB Error: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.get("/")
def read_root():
    return {"status": "online", "message": "API is running"}

@app.get("/api/devices")
def get_devices():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM devices ORDER BY brand, model;")
        devices = cursor.fetchall()
        return devices
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.get("/api/history/{device_id}")
def get_device_history(device_id: int):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM devices WHERE device_id = %s;", (device_id,))
        device = cursor.fetchone()
        cursor.execute("SELECT record_id, date_recorded, condition, resale_price, data_source FROM historical_prices WHERE device_id = %s ORDER BY date_recorded ASC;", (device_id,))
        history = cursor.fetchall()
        return {"device": device, "price_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.get("/api/predict/{device_id}")
def predict_price(device_id: int):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT date_recorded, resale_price FROM historical_prices WHERE device_id = %s ORDER BY date_recorded ASC;", (device_id,))
        records = cursor.fetchall()
        
        if len(records) < 5: 
            return {"predictions": []}
        
        df = pd.DataFrame(records)
        df['date_recorded'] = pd.to_datetime(df['date_recorded'])
        first_date = df['date_recorded'].min()
        df['days_passed'] = (df['date_recorded'] - first_date).dt.days
        
        X = df[['days_passed']]
        y = df['resale_price']
        model = LinearRegression().fit(X, y)
        
        last_date = df['date_recorded'].max()
        last_days_passed = df['days_passed'].max()
        
        predictions = []
        for days in [30, 90]:
            future_X = pd.DataFrame({'days_passed': [last_days_passed + days]})
            predicted_price = model.predict(future_X)[0]
            predictions.append({
                "days_out": days,
                "target_date": (last_date + timedelta(days=days)).strftime("%Y-%m-%d"),
                "estimated_value": round(max(predicted_price, 0), 2)
            })
            
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor: cursor.close()
        if conn: conn.close()