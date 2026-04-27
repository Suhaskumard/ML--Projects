def explain_result(prediction):
    if prediction == 1:
        return 'This job posting may be fraudulent. Verify company details.'
    return 'This job posting appears legitimate.'

