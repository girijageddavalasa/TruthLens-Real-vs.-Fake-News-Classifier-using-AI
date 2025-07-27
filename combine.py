import pandas as pd

# Load the datasets
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

# Add labels
true['label'] = 0
fake['label'] = 1

# Combine
data = pd.concat([true, fake], ignore_index=True)

# Shuffle and reset index
data = data.sample(frac=1).reset_index(drop=True)

# Save
data.to_csv("news.csv", index=False)
print("Combined and saved as news.csv")
