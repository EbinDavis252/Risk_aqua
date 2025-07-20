
-- Farmer Profile Table
CREATE TABLE IF NOT EXISTS farmer_profiles (
    farmer_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    region TEXT,
    income_per_month REAL,
    credit_score REAL,
    insurance_covered BOOLEAN
);

-- Loan Data Table
CREATE TABLE IF NOT EXISTS loan_data (
    loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer_id INTEGER,
    loan_amount REAL,
    loan_term_months INTEGER,
    emi_paid_on_time INTEGER,
    emi_missed INTEGER,
    last_payment_date DATE,
    defaulted INTEGER,
    FOREIGN KEY (farmer_id) REFERENCES farmer_profiles(farmer_id)
);

-- Sensor Data Table
CREATE TABLE IF NOT EXISTS farm_sensor_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME,
    farmer_id INTEGER,
    water_temp REAL,
    ph_level REAL,
    dissolved_oxygen REAL,
    ammonia_level REAL,
    mortality_reported INTEGER,
    farm_failure INTEGER,
    FOREIGN KEY (farmer_id) REFERENCES farmer_profiles(farmer_id)
);
