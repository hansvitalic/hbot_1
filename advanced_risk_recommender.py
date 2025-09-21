import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# 1. Load data
df = pd.read_csv('network_infrastructure_waterfall_risk_register.csv')

# 2. Encode categorical columns
categorical_cols = ['Risk Category', 'Risk Type', 'Severity', 'Likelihood', 'Status', 'RMS Step']
encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col + '_enc'] = encoders[col].fit_transform(df[col])

# 3. Prepare feature matrix
feature_cols = [col + '_enc' for col in categorical_cols] + ['Impact Score']
X = df[feature_cols]
# For recommenders, we use recommended action as the 'item' to suggest
actions = df['Recommended Action']

# 4. Fit a Nearest Neighbors model (content-based recommendation)
nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
nn.fit(X)

def recommend_action(risk_dict):
    """ 
    Given a dictionary of risk features, recommend top actions from similar risks
    Example risk_dict:
    {
        'Risk Category': 'Technical',
        'Risk Type': 'Software Bug',
        'Severity': 'High',
        'Likelihood': 'Medium',
        'Impact Score': 8,
        'Status': 'Open',
        'RMS Step': 'Analysis'
    }
    """
    # Encode input
    query = []
    for col in categorical_cols:
        val = risk_dict[col]
        if val not in encoders[col].classes_:
            # If unseen label, default to first class
            val = encoders[col].classes_[0]
        query.append(encoders[col].transform([val])[0])
    query.append(risk_dict['Impact Score'])
    # Find nearest risks
    distances, indices = nn.kneighbors([query])
    recommended_actions = actions.iloc[indices[0]].values
    return recommended_actions

# 5. Example usage
sample_risk = {
    'Risk Category': 'Technical',
    'Risk Type': 'Software Bug',
    'Severity': 'High',
    'Likelihood': 'Medium',
    'Impact Score': 8,
    'Status': 'Open',
    'RMS Step': 'Analysis'
}
print("Recommended actions for sample risk:")
for action in recommend_action(sample_risk):
    print("-", action)

# 6. Data exploration: Show top N actions by frequency
print("\nTop recommended actions by frequency:")
print(actions.value_counts().head(10))

# 7. Save encoder mappings for reproducibility
for col in categorical_cols:
    mapping = dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))
    print(f"\nEncoder mapping for {col}:")
    print(mapping)
