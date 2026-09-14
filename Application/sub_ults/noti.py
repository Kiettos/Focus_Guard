def get_notification(predictions: dict, threshold=2):
    """
    predictions = {"Engagement": 1, "Boredom": 2, "Confusion": 0, "Frustration": 1}
    """
    winner_label = max(predictions, key=predictions.get)
    winner_level = predictions[winner_label]
    
    if winner_level < threshold:
        return None  # không đủ mạnh để thông báo
    
    df = message_tables[winner_label]  # đã load sẵn 4 CSV vào dict lúc khởi động
    row = df[df["level"] == winner_level].iloc[0]
    return row["message"]