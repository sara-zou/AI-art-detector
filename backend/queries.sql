-- name: save_prediction
INSERT INTO predictions (user_id, image_name, image_path, label, confidence, prob_ai)
VALUES (?, ?, ?, ?, ?, ?);
 
-- name: get_predictions
SELECT id, image_name, image_path, label, confidence, prob_ai, created_at
FROM predictions
WHERE user_id = ?
ORDER BY created_at DESC;