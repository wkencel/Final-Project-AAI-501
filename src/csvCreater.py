# creates the csv for the metrics of the different models
# ONLY NEEDS TO BE RUN ONCE TO CREATE CSV FILE

# Sample code to create the initial CSV file with headers
import pandas as pd

headers = ['model', 'loss', 'accuracy', 'precision', 'f1_score'] + \
          ['accuracy_for_glass', 'accuracy_for_metal', 'accuracy_for_cardboard', 'accuracy_for_plastic']  # Add more classes as needed

# Create an empty DataFrame with these headers
df = pd.DataFrame(columns=headers)

# Save the empty DataFrame to a CSV file
df.to_csv('./model_metrics.csv', index=False)

print("CSV file with headers created.")