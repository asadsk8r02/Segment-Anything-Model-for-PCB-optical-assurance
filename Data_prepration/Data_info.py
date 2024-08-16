import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

folder_path = '/Users/asad/Documents/pcbdata/smd_annotation'

empty_count = 0
filled_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        empty_count += df['Designator'].isnull().sum()
        filled_count += df['Designator'].notnull().sum()

print("Number of empty entries in Designator column:", empty_count)
print("Number of filled entries in Designator column:", filled_count)

labels = ['Empty Entries', 'Filled Entries']
sizes = [empty_count, filled_count]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0) 
# plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.title('Empty vs Filled Entries in Designator Column')
# plt.show()


folder_path = "/Users/asad/Documents/pcbdata/new_merged_removed_final"
all_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

dfs = []
for file in all_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

designator_counts = combined_df['Designator'].value_counts()

unique_designators = len(designator_counts)

# print("Count for each unique designator:")
# print(designator_counts)
# print("\nTotal number of unique designators:", unique_designators)

# palette = sns.color_palette("husl", unique_designators)

# plt.figure(figsize=(12, 6))
# for i, (designator, count) in enumerate(designator_counts.items()):
#     plt.bar(designator, count, color=palette[i])  

# plt.title('Count of Unique Designators')
# plt.xlabel('Designator')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

folder_path = "/Users/asad/Documents/pcbdata/new_merged_removed_final"
all_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

dfs = []
for file in all_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

designator_counts = combined_df['Designator'].value_counts()

unique_designators = len(designator_counts)
# palette = sns.color_palette("husl", unique_designators)

# plt.figure(figsize=(12, 6))
# for i, (designator, count) in enumerate(designator_counts.items()):
#     plt.bar(designator, count, color=palette[i])  

# plt.title('Count of Unique Designators')
# plt.xlabel('Designator')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.yscale('log') 
# plt.tight_layout()
# plt.show()
