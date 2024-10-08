### Splitting each csv file into 2 different files. One with designator entries and the second without designator+Notes entries.
import os
import csv
import shutil
import os
import pandas as pd
import re

metadata_folder = "/Users/asad/Documents/PCB_files/ETL_pipelines/smd_annotation"

result_folder_with_designator = "/Users/asad/Documents/PCB_files/ETL_pipelines/w_d"
result_folder_without_designator = "/Users/asad/Documents/PCB_files/ETL_pipelines/wo_d"

os.makedirs(result_folder_with_designator, exist_ok=True)
os.makedirs(result_folder_without_designator, exist_ok=True)

for filename in os.listdir(metadata_folder):
    if filename.endswith(".csv"):
        metadata_filepath = os.path.join(metadata_folder, filename)

        with open(metadata_filepath, "r") as metadata_file:
            csv_reader = csv.DictReader(metadata_file)
            designator_present_rows = []
            designator_missing_rows = []

            for row in csv_reader:
                if row["Designator"]:
                    designator_present_rows.append(row)
                else:
                    designator_missing_rows.append(row)

        result_filename_with_designator = filename  
        result_filepath_with_designator = os.path.join(result_folder_with_designator, result_filename_with_designator)
        with open(result_filepath_with_designator, "w", newline="") as result_file:
            csv_writer = csv.DictWriter(result_file, fieldnames=csv_reader.fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(designator_present_rows)

        result_filename_without_designator = filename  
        result_filepath_without_designator = os.path.join(result_folder_without_designator, result_filename_without_designator)
        with open(result_filepath_without_designator, "w", newline="") as result_file:
            csv_writer = csv.DictWriter(result_file, fieldnames=csv_reader.fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(designator_missing_rows)

#### After this merged both files with and without designator+Notes into one single csv file as given in the starting.
#### The seperation in previous step was done to visually see how many files turn out to be empty without designator and how much data are we dealing with at the end.

def merge_and_sort_files(folder1, folder2, output_folder):
    files_folder1 = os.listdir(folder1)
    files_folder2 = os.listdir(folder2)

    common_files = list(set(files_folder1).intersection(files_folder2))

    for file_name in common_files:
        file_path1 = os.path.join(folder1, file_name)
        file_path2 = os.path.join(folder2, file_name)

        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)

        merged_df = pd.concat([df1, df2])

        merged_df = merged_df.sort_values('Instance ID')

        output_path = os.path.join(output_folder, file_name)

        merged_df.to_csv(output_path, index=False)
        
    # print("Files merged and sorted successfully!")

folder1= "/Users/asad/Documents/PCB_files/ETL_pipelines/w_d"
folder2= "/Users/asad/Documents/PCB_files/ETL_pipelines/wo_d"
output_folder= "/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged"

merge_and_sort_files(folder1, folder2, output_folder)

### After getting the merged files, the rows without designator entreis were removed as we could not get labels for them.

def remove_rows_without_designator(input_folder, output_folder):
    files = os.listdir(input_folder)

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        df = pd.read_csv(input_path)

        df = df.dropna(subset=['Designator'])

        df.to_csv(output_path, index=False)
        
    # print("Rows removed successfully!")

input_folder = "/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged"
output_folder = "/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged_removed"

remove_rows_without_designator(input_folder, output_folder)

### Perfom operations on newly merged SMD annotation files
### Performed designator label replacement operation through python.
### MANUALLY changed few designator labels as they were exceptions due to annotation errors. (Spelling mistakes or null values/unknown designators.)

folder_path = "/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged_removed"

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(file_path)

        df['Notes'] = df['Notes'].astype(str).str.lower()

        df['Notes'] = df['Notes'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        df['Notes'] = df['Notes'].str.replace('class', '')

        df['Notes'] = df['Notes'].str.replace('unsure', '')

        df['Notes'] = df['Notes'].str.strip()
        df['Notes'] = df['Notes'].str.replace(' ', '')

        df.to_csv(file_path, index=False)

        # print(f"Modified file saved: {file_path.strip()}")

csv_files = glob.glob("/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged_removed/*.csv")

for file in csv_files:

    df = pd.read_csv(file)

    df.loc[df['Notes'] == 'connector', 'Designator'] = 'A'
    df.loc[df['Notes'] == 'header', 'Designator'] = 'H'
    df.loc[df['Notes'] == 'button', 'Designator'] = 'BTN'
    df.loc[df['Notes'] == 'diode', 'Designator'] = 'D'
    df.loc[df['Notes'] == 'led', 'Designator'] = 'LED'
    #df.loc[df['Notes'].isin(['led', 'lled']), 'Designator'] = 'LED'
    df.loc[df['Notes'] == 'fuse', 'Designator'] = 'F'
    df.loc[df['Notes'] == 'socket', 'Designator'] = 'SC'
    df.loc[df['Notes'] == 'jack', 'Designator'] = 'J'
    df.loc[df['Notes'] == 'jackdevicetextsticker', 'Designator'] = 'J'    
    df.loc[df['Notes'] == 'switch', 'Designator'] = 'SW'
    df.loc[df['Notes'] == 'pin', 'Designator'] = 'E'
    df.loc[df['Notes'] == 'jumper', 'Designator'] = 'JP'
    df.loc[df['Notes'] == 'relay', 'Designator'] = 'K'
    df.loc[df['Notes'] == 'crystaloscillator', 'Designator'] = 'CO'
    df.loc[df['Notes'] == 'capristor', 'Designator'] = 'C'
    df.loc[df['Notes'] == 'zenerdiode', 'Designator'] = 'ZD'
    df.loc[df['Notes'] == 'resistorarray', 'Designator'] = 'RN'
        
    df.loc[(df['Designator'] == 'B') & (df['Notes'].isnull()), 'Designator'] = 'R'
    df.loc[(df['Designator'] == 'M') & (df['Notes'].isnull()), 'Designator'] = 'IC'
    df.loc[(df['Designator'] == 'CB') & (df['Notes'].isnull()), 'Designator'] = 'C'
    df.loc[(df['Designator'] == 'CBP') & (df['Notes'].isnull()), 'Designator'] = 'C'
    df.loc[(df['Designator'] == 'CN') & (df['Notes'].isnull()), 'Designator'] = 'A'
    df.loc[(df['Designator'] == 'COM') & (df['Notes'].isnull()), 'Designator'] = 'A'
    df.loc[(df['Designator'] == 'CON') & (df['Notes'].isnull()), 'Designator'] = 'A'
    df.loc[(df['Designator'] == 'X') & (df['Notes'].isnull()), 'Designator'] = 'CO'
    df.loc[(df['Designator'] == 'Y') & (df['Notes'].isnull()), 'Designator'] = 'CO'
    df.loc[(df['Designator'] == 'U') & (df['Notes'].isnull()), 'Designator'] = 'IC'
    df.loc[(df['Designator'] == 'REG') & (df['Notes'].isnull()), 'Designator'] = 'VR'
    df.loc[(df['Designator'] == 'RA') & (df['Notes'].isnull()), 'Designator'] = 'RN'
    df.loc[(df['Designator'] == 'Q') & (df['Notes'].isnull()), 'Designator'] = 'TR'
    df.loc[(df['Designator'] == 'QF') & (df['Notes'].isnull()), 'Designator'] = 'TR'
    df.loc[(df['Designator'] == 'QM') & (df['Notes'].isnull()), 'Designator'] = 'TR'
    df.loc[(df['Designator'] == 'P') & (df['Notes'].isnull()), 'Designator'] = 'PL'
    df.loc[(df['Designator'] == 'PR') & (df['Notes'].isnull()), 'Designator'] = 'RN'
    df.loc[(df['Designator'] == 'MR') & (df['Notes'].isnull()), 'Designator'] = 'RN'
    df.loc[(df['Designator'] == 'MSP') & (df['Notes'].isnull()), 'Designator'] = 'IC'
    
    df = df.loc[~((df['Designator'] == 'U') & (df['Notes'] == 'devicetextsticker'))]

    df.to_csv(file, index=False)


# Results written to new csv result file
folder_path = "/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged_removed"

designator_notes_frequency_dict = {}  
designator_file_dict = {} 

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            designator = str(row['Designator'])
            notes = str(row['Notes'])
            designator_notes_pair = (designator, notes)
            
            if designator_notes_pair in designator_notes_frequency_dict:
                designator_notes_frequency_dict[designator_notes_pair] += 1
                designator_file_dict[designator_notes_pair].append(file_name)
            else:
                designator_notes_frequency_dict[designator_notes_pair] = 1
                designator_file_dict[designator_notes_pair] = [file_name]

sorted_data = sorted(designator_notes_frequency_dict.items(), key=lambda x: x[0][0])
table_data = [(designator, notes, frequency, ", ".join(designator_file_dict[(designator, notes)])) for ((designator, notes), frequency) in sorted_data]

df_output = pd.DataFrame(table_data, columns=['Designator', 'Notes', 'Frequency', 'File Names'])

output_csv_file = "/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged_removed/output_table_data_manual_operation.csv"
df_output.to_csv(output_csv_file, index=False)
# print(f"CSV table written to {output_csv_file}")


# folder_path = "/Users/asad/Documents/PCB_files/ETL_pipelines/new_merged_removed"

# unique_designators = set()  


# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.csv'):
#         file_path = os.path.join(folder_path, file_name)

#         df = pd.read_csv(file_path)
        
#         unique_designators.update(df['Designator'].dropna().unique())

# sorted_designators = sorted(unique_designators)
# table_data = [(i+1, designator) for i, designator in enumerate(sorted_designators)]
# print(tabulate(table_data, headers=['Index', 'Designator'], tablefmt='grid'))
