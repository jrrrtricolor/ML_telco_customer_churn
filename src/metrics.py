from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions made')
ERRORS_TOTAL = Counter('api_errors_total', 'Total number of errors encountered on api')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Duration of prediction processing in seconds')
AVG_CONFIDENCE = Histogram('average_prediction_confidence', 'Average confidence of predictions')
