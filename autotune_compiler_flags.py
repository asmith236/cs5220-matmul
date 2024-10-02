# this program tests the performance of all combinations of ging compiler flags and outputs the flag combination resulting in the best performance
# python3 autotune_compiler_flags.py

import subprocess
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import csv

# makefile to be modified
makefile_in = "Makefile.in.gcc"
makefile_backup = "Makefile.in.gcc.bak"

# output CSV file for results
output_csv = "autotune_compiler_flags_out.csv"

# gcc flags to test
opt_flags = [
    '-O3', '-march=native', '-ftree-vectorize', '-funroll-loops', '-ffast-math', '-fprefetch-loop-arrays', '-ftree-loop-distribute-patterns', '-floop-interchange', '-floop-strip-mine'
]

# fn to update makefile.in.gcc with the current flags
def update_makefile(flags):
    with open(makefile_in, 'r') as file:
        content = file.read()

    # replace the OPTFLAGS line
    content = re.sub(r"OPTFLAGS = .+", f"OPTFLAGS = {flags}", content)

    with open(makefile_in, 'w') as file:
        file.write(content)

def build_and_run():
    # compile and run the dgemm_mine.c

    # run 'make realclean' before building to ensure a clean state
    subprocess.run(["make", "realclean"], capture_output=True)

    # build the program using make
    build_process = subprocess.run(["make"], capture_output=True, text=True)
    if build_process.returncode != 0:
        print("Build failed!")
        return None
    
    # run the executable
    run_process = subprocess.run(["./matmul-mine"], capture_output=True, text=True)
    if run_process.returncode != 0:
        print("Execution failed!")
        return None
    
    # read and parse the timing-mine.csv file
    try:
        df = pd.read_csv("timing-mine.csv")
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

def append_results_to_file(flags, results):
    # append the results for a specific block size to the output csv file
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for size, mflop in results.items():
            writer.writerow([flags, size, mflop])

def plot(df, filename):
    # plot the results for each flag combination
    plt.figure(figsize=(12, 8))
    for flags, group in df.groupby('Flags'):
        plt.plot(group['Matrix Size'], group['MFLOP'], label=flags)

    plt.title("Matrix Size vs. MFLOP for Different GCC Flag Combinations")
    plt.xlabel("Matrix Size")
    plt.ylabel("MFLOP (Million Floating Point Operations Per Second)")
    lgd = plt.legend(title="GCC Flags", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

# save a backup of the original makefile
if not os.path.exists(makefile_backup):
    subprocess.run(["cp", makefile_in, makefile_backup])

# init the output csv file with headers
if not os.path.exists(output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Flags", "Matrix Size", "MFLOP"])

all_results = {}

# baseline test (no flags)
print("testing baseline (no flags):")
update_makefile("")
baseline_results = build_and_run()
if baseline_results is None:
    print("baseline test failed. exiting.")
    subprocess.run(["mv", makefile_backup, makefile_in])
    exit(1)

baseline_average = sum(baseline_results.values()) / len(baseline_results)
print(f"baseline average mflop: {baseline_average:.2f}")

# test each flag individually
improved_flags = []
for flag in opt_flags:
    print(f"testing individual flag: {flag}")
    update_makefile(flag)
    current_results = build_and_run()
    if current_results is None:
        print(f'^^ {flag}')
        continue
    
    # calc the average mflop for this flag
    average_mflop = sum(current_results.values()) / len(current_results)
    print(f"avg mflop with {flag}: {average_mflop:.2f}")
    
    append_results_to_file(flag, current_results)
    
    # check if this flag improves performance over the baseline
    if average_mflop > baseline_average:
        improved_flags.append(flag)

# test combinations of improved flags
print(f"flags that improved performance: {improved_flags}")
if len(improved_flags) > 1:
    for i in range(2, len(improved_flags) + 1): # start from 2 to exclude single flags
        for combo in combinations(improved_flags, i):
            flags = " ".join(combo)
            print(f"testing combination: {flags}")
            
            update_makefile(flags)
            current_results = build_and_run()
            if current_results is None:
                print(f'^^ {flags}')
                continue

            # calc the average mflop for current combo of flags
            average_mflop = sum(current_results.values()) / len(current_results)
            print(f"average MFLOP with {flags}: {average_mflop:.2f}")
            
            # append results to the file
            append_results_to_file(flags, current_results)
else:
    print("no individual flags improved performance over baseline.")

# restore the original Makefile
subprocess.run(["mv", makefile_backup, makefile_in])

# create a df from the results
df = pd.read_csv(output_csv)

# calc the average performance for each combination
average_performance = df.groupby('Flags')['MFLOP'].mean().sort_values(ascending=False)
best_flags = average_performance.idxmax()
best_average_mflop = average_performance.max()

# output the best flag combination
print(f"\nbest gcc flag combination: {best_flags} with average mflop = {best_average_mflop:.2f}")

top_5_flag_combos = average_performance.nlargest(5)
print("top 5 block sizes with highest avg mflop performance:")
print(top_5_flag_combos)

top_5_flag_combos_list = top_5_flag_combos.index.tolist()
df_top_5 = df[df['Flags'].isin(top_5_flag_combos_list)]
plot(df_top_5, "autotune_compiler_flags_out.pdf")
