
# this program run the dgemm_mine.c implementation on multiple block sizes, and outputs a graph with the performance of the top 5 block sizes
# execute by running python3 autotune_block_size.py

import subprocess
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

# c source file
source_file = "dgemm_mine.c"
output_csv = "autotune_block_size_out.csv"

# define the range of block sizes to test
block_sizes = range(24, 256, 8)  

all_results = {}

def update_block_size(size):
    # update BLOCK_SIZE in dgemm_mine.c
    with open(source_file, 'r') as file:
        content = file.read()
    
    content = re.sub(r"#define BLOCK_SIZE \(\(int\) \d+\)", f"#define BLOCK_SIZE ((int) {size})", content)
    
    with open(source_file, 'w') as file:
        file.write(content)

def build_and_run():
    # compile and run dgemm_mine.c
    build_process = subprocess.run(["make"], capture_output=True, text=True)
    if build_process.returncode != 0:
        print("Build failed!")
        return None
    
    # gen the timing-mine.csv file
    run_process = subprocess.run(["./matmul-mine"], capture_output=True, text=True)
    if run_process.returncode != 0:
        print("Execution failed!")
        return None
    
    # read and parse the timing-mine.csv file
    try:
        # read the csv file into a pandas DataFrame
        df = pd.read_csv("timing-mine.csv")
        
        # conv df to a dictionary where size is the key and mflop is the value
        results = df.set_index('size')['mflop'].to_dict()
        
        return results
    except FileNotFoundError:
        print("timing-mine.csv not found. Make sure the program generated the file.")
        return None
    except pd.errors.EmptyDataError:
        print("timing-mine.csv is empty. No data to process.")
        return None
    except pd.errors.ParserError:
        print("timing-mine.csv is malformed. Could not parse the data.")
        return None

def append_results_to_file(block_size, results):
    # append the results for a specific block size to the output csv file
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for size, mflop in results.items():
            writer.writerow([block_size, size, mflop])

def plot(df, filename):
    plt.figure(figsize=(12, 8))
    # group data by block size and plot each group
    for block_size, group in df.groupby('Block Size'):
        plt.plot(group['Matrix Size'], group['MFLOP'], label=f"Block Size {block_size}")

    # add labels and customizations
    plt.title("Performance of Top 5 Performing Block Sizes")
    plt.xlabel("Matrix Size")
    plt.ylabel("MFLOP")
    lgd = plt.legend(title="Block Size")
    plt.grid(True)
    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

# init output file with headers
if os.path.exists(output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Block Size", "Matrix Size", "MFLOP"])

# run matmul-mine for each block size
for size in block_sizes:
    print(f"testing BLOCK_SIZE = {size}")
    
    update_block_size(size)
    current_results = build_and_run()
    if current_results is None:
        print("error")
        continue
    
    # add results to output csv
    all_results[size] = current_results
    append_results_to_file(size, current_results)

df = pd.read_csv("autotune_block_size_out.csv")

# group the data by block size and calculate the average mflop for each block size
average_mflop = df.groupby('Block Size')['MFLOP'].mean()

# sort the block sizes by average mflop in descending order and select the top 5
top_5_block_sizes = average_mflop.nlargest(5)

print("top 5 block sizes with highest avg mflop performance:")
print(top_5_block_sizes)

top_5_block_sizes_list = top_5_block_sizes.index.tolist()
print("\nblock sizes with the highest avg performance:", top_5_block_sizes_list)

df_top_5 = df[df['Block Size'].isin(top_5_block_sizes_list)]
plot(df_top_5, "autotune_block_size_out.pdf")
