from sklearn import metrics

def score_function(y_batch, y_pred):
    actual = y_batch.reshape(y_pred.shape)
    predicted = y_pred
    mape = metrics.mean_absolute_percentage_error(actual, predicted)
    score = max(0, 100 * (1 - mape))
    return score