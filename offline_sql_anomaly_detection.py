import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import generate_gitlab_malicious_queries


def normalize_query(sql_statement):
    normalized = re.sub(r'\b\d+\b', '?', sql_statement)  # Replace numbers
    normalized = re.sub(r"'[^']*'", "?", normalized)  # Replace single-quoted strings
    normalized = re.sub(r'"[^"]*"', "?", normalized)  # Replace double-quoted strings
    return normalized


# feature extraction
def feature_extraction_with_normalization(sql_statements, frequencies):
    features = pd.DataFrame()
    features['query_length'] = sql_statements.apply(len)  # query length
    features['structure_frequency'] = frequencies  # frequency of structure
    features['num_special_chars'] = sql_statements.apply(lambda x: len(re.findall(r'[\'\";#]', x)))  # special chars
    features['num_operations'] = sql_statements.apply(lambda x: x.count(';'))  # # of SQL operations
    features['num_keywords'] = sql_statements.apply(
        lambda x: len(re.findall(r'\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|AND|OR|LIMIT)\b', x, re.IGNORECASE))
    )  # num SQL keywords
    return features


chunk_size = 10000
data_chunks = pd.read_csv('output_all.csv', chunksize=chunk_size)

processed_chunks = []
feature_chunks = []

for chunk in data_chunks:
    chunk['Normalized Query'] = chunk['SQL Statement'].apply(normalize_query)
    chunk_structure_freq = chunk['Normalized Query'].value_counts()
    chunk['Structure Frequency'] = chunk['Normalized Query'].map(chunk_structure_freq)
    processed_chunks.append(chunk)
    feature_chunks.append(
        feature_extraction_with_normalization(chunk['Normalized Query'], chunk['Structure Frequency'])
    )

processed_data = pd.concat(processed_chunks, ignore_index=True)
features_combined = pd.concat(feature_chunks, ignore_index=True)

# normalize features for training
scaler = StandardScaler()
features_combined_scaled = scaler.fit_transform(features_combined)

# model
model = IsolationForest(contamination=0.019, random_state=42)
model.fit(features_combined_scaled)

# get malicious queries, label, and prepare for testing by normalizing
malicious_queries = generate_gitlab_malicious_queries.generate_gitlab_malicious_statements_with_sqli()
malicious_queries_df = pd.DataFrame({'SQL Statement': malicious_queries, 'Label': ['anomaly'] * len(malicious_queries)})
malicious_queries_df['Normalized Query'] = malicious_queries_df['SQL Statement'].apply(normalize_query)
malicious_queries_df['Structure Frequency'] = malicious_queries_df['Normalized Query'].map(
    processed_data['Normalized Query'].value_counts()
).fillna(0)

# test dataset (500 benign, x anomalous)
test_data = pd.concat([processed_data.sample(500, random_state=42), malicious_queries_df], ignore_index=True)

# feature extract our test datac
test_features = feature_extraction_with_normalization(test_data['Normalized Query'], test_data['Structure Frequency'])
test_features_scaled = scaler.transform(test_features)

# predict
test_data['Anomaly'] = model.predict(test_features_scaled)

# Evaluate results
true_labels = test_data['Label'].map({'benign': 1, 'anomaly': -1}).fillna(1).astype(int)
conf_matrix = confusion_matrix(true_labels, test_data['Anomaly'])
class_report = classification_report(true_labels, test_data['Anomaly'], target_names=['Anomaly', 'Benign'])


