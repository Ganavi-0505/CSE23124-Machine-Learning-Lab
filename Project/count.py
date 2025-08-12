import pandas as pd

# Read your CSV file
df = pd.read_csv(r"combined.csv")   # path of your merged CSV

# Count how many times each model appears
model_counts_df = df['model'].value_counts().reset_index()
model_counts_df.columns = ['model', 'count']

# Print results
print(model_counts_df.to_string(index=False))

# Print total count
total_count = model_counts_df['count'].sum()
print("Total count:", total_count)

output_path = r"model_count.txt"  # change path if needed
with open(output_path, "w") as f:
    f.write(model_counts_df.to_string(index=False))
    f.write(f"\nTotal count: {total_count}\n")

print(f"Results saved to: {output_path}")
