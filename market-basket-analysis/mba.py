import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# ==============================
# LOAD DATA
# ==============================
import kagglehub
path = kagglehub.dataset_download("heeraldedhia/groceries-dataset")
df = pd.read_csv(f"{path}/Groceries_dataset.csv")
print("Dataset Loaded Successfully\n")

# ==============================
# PREPROCESSING
# ==============================
transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)
transactions = transactions.tolist()

# ==============================
# ONE-HOT ENCODING
# ==============================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_array, columns=te.columns_)

print("Basket Shape:", basket.shape)

# ==============================
# APRIORI
# ==============================
start = time.time()
apriori_items = apriori(basket, min_support=0.01, use_colnames=True)
apriori_time = time.time() - start

apriori_rules = association_rules(apriori_items, metric="lift", min_threshold=1)

# ==============================
# FP-GROWTH
# ==============================
start = time.time()
fp_items = fpgrowth(basket, min_support=0.01, use_colnames=True)
fp_time = time.time() - start

fp_rules = association_rules(fp_items, metric="lift", min_threshold=1)

# ==============================
# RULE RANKING
# ==============================
apriori_rules['score'] = apriori_rules['confidence'] * apriori_rules['lift']
top_rules = apriori_rules.sort_values(by='score', ascending=False)

print("\nTop Rules:\n")
print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

# ==============================
# STRONG RULES
# ==============================
strong_rules = apriori_rules[
    (apriori_rules['confidence'] > 0.3) &
    (apriori_rules['lift'] > 1.2)
]

# ==============================
# RECOMMENDATION SYSTEM
# ==============================
def recommend_products(product, rules, top_n=5):
    recommendations = []

    for _, row in rules.iterrows():
        if product in row['antecedents']:
            recommendations.append((list(row['consequents']), row['confidence']))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

print("\nRecommendations for 'whole milk':")
print(recommend_products('whole milk', apriori_rules))

# ==============================
# VISUALIZATION
# ==============================

# Scatter Plot
plt.figure()
plt.scatter(apriori_rules['support'], apriori_rules['confidence'])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
plt.show()

# Top Products
top_items = df['itemDescription'].value_counts().head(10)
plt.figure()
top_items.plot(kind='bar')
plt.title("Top Selling Products")
plt.show()

# Heatmap (simplified)
pivot = apriori_rules.pivot_table(
    index='antecedents',
    columns='consequents',
    values='confidence'
)

plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="coolwarm")
plt.title("Confidence Heatmap")
plt.show()

# ==============================
# PERFORMANCE COMPARISON
# ==============================
print("\nExecution Time:")
print("Apriori:", apriori_time)
print("FP-Growth:", fp_time)

# ==============================
# BUSINESS INSIGHTS
# ==============================
print("\nBusiness Insights:\n")
for _, row in strong_rules.head(5).iterrows():
    print(f"If customer buys {list(row['antecedents'])}, "
          f"they also buy {list(row['consequents'])} "
          f"(confidence: {round(row['confidence'],2)})")
