def explain(features):
    reasons=[]
    if features['urgent_words']:
        reasons.append('Contains urgency/scam style wording')
    if features['link_count']>2:
        reasons.append('Too many external links')
    return reasons or ['No major fraud indicators found']

