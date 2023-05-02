from sklearn import metrics

def score_function(y_batch, y_pred):
    # actual = y_batch.reshape(y_pred.shape)
    actual = y_batch
    predicted = y_pred
    mape = metrics.mean_absolute_percentage_error(actual, predicted)
    score = max(0, 100 * (1 - mape/100))
    return score