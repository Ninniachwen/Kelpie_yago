# extract input facts
import copy
from inspect import getargvalues, stack
import os
from pathlib import Path
import time
import pandas as pd

from config import FACTS_PATH
from config import DATA_PATH


def extract_input_facts(source_file:str, output_file:str, seeds:list[int]=[42], min_rank:int=3, mode:str="RND", n:int=100, replace:bool=False, direction:str="tail"):
    """
    extracts n lines from the raw file of "filtered_ranks.csv". 
    Mode RND extracts them randomly, depending on the seed and writes as many files as seeds are given. 
    Mode TOP returns the top n lines, sorted by tail/head rank.
    if there are less than n predictions with rank 1, then the outputfile is smaller. reducing the min_rank to 3 for example, gives access to more "good" predictions to use.

    Args:
        source_file (str): file with table of predictions and rank, usually "{dataset}/filtered_ranks.csv"
        output_file (str): filename to write "input_facts" to. eg. input_facts/transe_{dataset}.csv
        seeds (list, optional): list of seeds for random mode, as many input_fact files as seeds. Defaults to [42].
        min_rank (int, optional): minimum rank of predictions which are considered for selection. rank 1 are correct predictions, if there are less than n of these, rank should be reduced. Defaults to 3.
        mode (str, optional): random selection or top n lines. Defaults to "RND".
        replace (bool, optional): false is recommended, this prevents duplicate lines in the output file. Defaults to False.
        direction (str, optional): "head" or "tail" prediction. Defaults to "tail".

    returns:
        last file name of created input_facts
    """ 
    args = arguments()

    if type(seeds) == int:
        seeds = [seeds]
    
    if (len(seeds)>1) & (mode=="TOP"):
        print("INFO: caution, giving several seeds doesn't change the output of top_n rows. only one file will be created")

    input_path = os.path.join(DATA_PATH, source_file)
    facts_path = os.path.join(FACTS_PATH, output_file)
    output_prefix, timestamp = generate_output_file_prefix()

    df = pd.read_csv(input_path, sep=';', header=0, names=["s", "p", "o", "head_rank", "tail_rank"])
    
    df_filtered = df[df[f"{direction}_rank"] <= min_rank]
    
    for seed in seeds:
        if mode == "RND":
            df_subset = df_filtered.sample(n=n, replace=replace, random_state=seed) #randomly select n rows. with or without repeat
            df_subset = df_subset.iloc[:, :3]
        elif mode == "TOP":
            df_subset = df_subset.sort_values(by=f"{direction}_rank", ascending=[True])
            df_subset = df_filtered.iloc[:n, :3]

        # output files
        # data
        filename = f"{facts_path}_{direction}_{seed}.csv"
        df_subset.to_csv(filename, sep='\t', index=False, header=False)
        
        # data logging
        filename_7_output = f"{output_prefix}_7_filtered_ranks_{direction}_{seed}.csv"
        df_subset.to_csv(filename_7_output, sep='\t', index=False, header=False)

    #logging
    filename_6_output = f"{output_prefix}__6_input_facts_logging.txt"
    args["output_path"] = filename
    with open(filename_6_output, "w") as execution_log:
        execution_log.write(f"test.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")

    print(f"{direction} length: {len(df_filtered)}, sample: {len(df_subset)}")
    print(filename_7_output)
    return filename_7_output

@staticmethod
def arguments():
    """Returns dictionary of calling function's
        named arguments, that existed at that time.
    """
    args = getargvalues(stack()[1][0])[-1:]
    return copy.deepcopy(args[0])

@staticmethod
def generate_output_file_prefix(mode="necessary"):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    day_folder = time.strftime("%Y-%m-%d")
    path = os.path.join("outputs", day_folder)
    Path(path).mkdir(parents=True, exist_ok=True)
    output_prefix = os.path.join(path, timestamp + f"_transE_{mode[0]}")
    return output_prefix, timestamp
    
@staticmethod
def flatten_list(list_of_lists:list[list[tuple[int, int, int]]])-> list[tuple[int, int, int]]:
    flat_list = []
    for li in list_of_lists:
        for triple in li:
            flat_list.append(triple)
    return flat_list

if __name__ == "__main__":
    dataset = "FR_Reduced_2K"
    filtered_ranks = f"{dataset}/TransE_filtered_ranks.csv"
    write_filename = f"TransE_{dataset}".lower()

    extract_input_facts(source_file=filtered_ranks, output_file=write_filename, min_rank=1, mode="RND", seeds=[42])