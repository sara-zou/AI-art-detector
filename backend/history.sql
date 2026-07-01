CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    image_name  TEXT,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    prob_ai     REAL NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);