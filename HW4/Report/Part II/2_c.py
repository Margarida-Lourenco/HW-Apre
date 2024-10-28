import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Load the data
data = pd.read_csv('accounts.csv')
df = data.iloc[:, :8]
df = df.drop_duplicates().dropna()

df["clusters"] = clusters

plt.figure(figsize=(12, 6))

# Plot for "job"
sns.displot(data=df, x="job", hue ="clusters", multiple="dodge", stat="density", shrink=0.8, common_norm=False)
plt.title('Frequency Distribution of Job')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Plot for "education"
sns.displot(data=df, x="education", hue ="clusters", multiple="dodge", stat="density", shrink=0.8, common_norm=False)
plt.title('Frequency Distribution of Education')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()