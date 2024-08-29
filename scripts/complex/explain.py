import sys
import os
import argparse
import random
import time
import numpy
import torch


sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from config import DATA_PATH, OUTPUT_PATH
from ruleset import EmptyResultException
from utils import arguments, generate_output_file_prefix
from dataset import ALL_DATASET_NAMES, Dataset
from kelpie import Kelpie as Kelpie
from data_poisoning import DataPoisoning
from criage import Criage
from link_prediction.models.complex import ComplEx
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, \
    BATCH_SIZE, REGULARIZER_NAME
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER


def main():
    datasets = ALL_DATASET_NAMES

    parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")


    optimizers = ['Adagrad', 'Adam', 'SGD']
    parser.add_argument('--optimizer',
                        choices=optimizers,
                        default='Adagrad',
                        help="Optimizer in {} to use in post-training".format(optimizers))

    parser.add_argument('--batch_size',
                        default=1000,
                        type=int,
                        help="Batch size to use in post-training")

    parser.add_argument('--max_epochs',
                        default=50,
                        type=int,
                        help="Number of epochs to run in post-training")

    parser.add_argument('--dimension',
                        default=1000,
                        type=int,
                        help="Factorization rank.")

    parser.add_argument('--learning_rate',
                        default=1e-1,
                        type=float,
                        help="Learning rate")

    parser.add_argument('--reg',
                        default=5e-3,
                        type=float,
                        help="Regularization weight")

    parser.add_argument('--init',
                        default=1e-3,
                        type=float,
                        help="Initial scale")

    parser.add_argument('--decay1',
                        default=0.9,
                        type=float,
                        help="Decay rate for the first moment estimate in Adam")

    parser.add_argument('--decay2',
                        default=0.999,
                        type=float,
                        help="Decay rate for second moment estimate in Adam")

    parser.add_argument("--coverage",
                        type=int,
                        default=10,
                        help="Number of random entities to extract and convert")

    parser.add_argument("--baseline",
                        type=str,
                        default=None,
                        choices=[None, "k1", "data_poisoning", "criage"],
                        help="attribute to use when we want to use a baseline rather than the Kelpie engine")

    parser.add_argument("--entities_to_convert",
                        type=str,
                        help="path of the file with the entities to convert (only used by baselines)")

    parser.add_argument("--mode",
                        type=str,
                        default="necessary",
                        choices=["sufficient", "necessary"],
                        help="The explanation mode")

    parser.add_argument("--relevance_threshold",
                        type=float,
                        default=None,
                        help="The relevance acceptance threshold to use")

    prefilters = [TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER]
    parser.add_argument('--prefilter',
                        choices=prefilters,
                        default=None,
                        help="Prefilter type in {} to use in pre-filtering".format(prefilters))

    parser.add_argument("--prefilter_threshold",
                        type=int,
                        default=20,
                        help="The number of promising training facts to keep after prefiltering")
        
    parser.add_argument("--verbose",
                        type=str,
                        default=True,
                        help="dertermines quantity of print output")

    parser.add_argument('--dataset',
                        choices=datasets,
                        default="FR_Reduced_2K",
                        help="Dataset in {}".format(datasets),
                        #required=True
                        )

    parser.add_argument('--model_path',
                        default="stored_models/ComplEx_FR_Reduced_2K.pt",
                        help="Path to the model to explain the predictions of",
                        #required=True
                        )

    parser.add_argument("--facts_to_explain_path",
                        type=str,
                        default="input_facts/complex_fr_reduced_2k_tail_42.csv",
                        #required=True,
                        help="path of the file with the facts to explain the predictions of.")
        
    parser.add_argument("--rules_file",
                        type=str,
                        #default="yago3-10_rules.csv",
                        default="fr_reduced_2k_rules.csv",
                        help="The rule file for symbolic filter")

    parser.add_argument("--second_rules_file",
                        type=str,
                        default="FR_editorial_rules.csv",
                        #default=None,
                        help="The rule file for symbolic filter")

    parser.add_argument("--builder",
                        type=str,
                        #default="rules_kelpie_reverse",
                        #default="rules_heuristic_pca",     #TODO: make list of constants
                        #default="rules_kelpie",
                        #default="rules_heuristic_frequency",
                        default="kelpie",
                        help="dertermines which builder and heuristic is used")

    args = parser.parse_args()

    # deterministic!
    seed = 42
    torch.backends.cudnn.deterministic = True
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    kernel = ""
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        kernel = "cuda"
    else:
        torch.set_rng_state(torch.get_rng_state())
        kernel = "cpu"

    if args.prefilter == None:
        if ("rules" in args.builder) or ("heuristic" in args.builder):
            prefilter = "symbolic_based"
        elif args.builder == "kelpie":
            prefilter = "topology_based"
        else:
            raise Exception("no valid builder on which to choose prefilter")
        
    args_dict = arguments()

    hyperparameters = {DIMENSION: args.dimension,
                    INIT_SCALE: args.init,
                    LEARNING_RATE: args.learning_rate,
                    OPTIMIZER_NAME: args.optimizer,
                    DECAY_1: args.decay1,
                    DECAY_2: args.decay2,
                    REGULARIZER_WEIGHT: args.reg,
                    EPOCHS: args.max_epochs,
                    BATCH_SIZE: args.batch_size,
                    REGULARIZER_NAME: "N3"}

    relevance_threshold = args.relevance_threshold
    rules_file = args.rules_file
    second_rules_file = args.second_rules_file
    verbose = args.verbose
    builder = args.builder
    dataset_name = args.dataset
    facts_to_explain_path = args.facts_to_explain_path

    

    ########## LOAD DATASET


    # load the dataset and its training samples
    print("Loading dataset %s..." % dataset_name)
    dataset = Dataset(name=dataset_name, separator="\t", load=True)

    output_prefix, timestamp = generate_output_file_prefix()
    filename_8_explain = output_prefix + "_08_explain_py.txt"
    print(filename_8_explain)
    filename_9_output = output_prefix + f"_09_{dataset.name}_output.csv"
    print(filename_9_output)
    with open(filename_8_explain, "w") as execution_log:
        execution_log.write(f"explain.py at {timestamp}\n")
        execution_log.write(str(args_dict)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}, output: {filename_9_output}\n")

    print(f"Reading facts to explain... {facts_to_explain_path}")
    with open(facts_to_explain_path, "r", encoding="utf-8") as facts_file:
        testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)  # type: ComplEx
    if torch.cuda.is_available():
        model.to('cuda')
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))) 
    model.eval()

    start_time = time.time()

    if args.baseline is None:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                            relevance_threshold=relevance_threshold, output_prefix=output_prefix, rules_file=rules_file,
                            rules_file2=second_rules_file, verbose=verbose, builder=builder)
    elif args.baseline == "data_poisoning":
        kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter)
    elif args.baseline == "criage":
        kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
    elif args.baseline == "k1":
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold, max_explanation_length=1)
    else:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                            relevance_threshold=relevance_threshold, output_prefix=output_prefix, rules_file=rules_file,
                            rules_file2=second_rules_file, verbose=verbose, builder=builder)

    testing_fact_2_entities_to_convert = None
    if args.mode == "sufficient" and args.entities_to_convert is not None:
        print("Reading entities to convert...")
        testing_fact_2_entities_to_convert = {}
        with open(args.entities_to_convert, "r") as entities_to_convert_file:
            entities_to_convert_lines = entities_to_convert_file.readlines()
            i = 0
            while i < len(entities_to_convert_lines):
                cur_head, cur_rel, cur_name = entities_to_convert_lines[i].strip().split(";")
                assert [cur_head, cur_rel, cur_name] in testing_facts
                cur_entities_to_convert = entities_to_convert_lines[i + 1].strip().split(",")
                testing_fact_2_entities_to_convert[(cur_head, cur_rel, cur_name)] = cur_entities_to_convert
                i += 3

    output_lines = []
    statistics = {"facts":0, "complexity_sum":0, "time":0, "avg_time":0, "avg_complexity":0, "empty":[], "l.599>xsi":[], "l.552>xsi":[], "l.526>xsi":[], "l.484>xsi":[], "no partial set":[]}    #TODO: remove debugging entries
    for i, fact in enumerate(testing_facts):
        try:
            head, relation, tail = fact
        except:                             # if row is empty or mal formatted
            if verbose: print(f"skipped explaining row '{', '.join(fact)}', prediction might be malformed or empty")   # print(*fact, sep = ", ")
            continue
        print("Explaining fact " + str(i) + " on " + str(
            len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
        head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                        dataset.get_id_for_relation_name(relation), \
                                        dataset.get_id_for_entity_name(tail)
        sample_to_explain = (head_id, relation_id, tail_id)

        if args.mode == "sufficient":
            entities_to_convert_ids = None if testing_fact_2_entities_to_convert is None \
                else [dataset.entity_name_2_id[x] for x in testing_fact_2_entities_to_convert[(head, relation, tail)]]

            rule_samples_with_relevance, \
            entities_to_convert_ids = kelpie.explain_sufficient(sample_to_explain=sample_to_explain,
                                                                perspective="head",
                                                                num_promising_samples=args.prefilter_threshold,
                                                                num_entities_to_convert=args.coverage,
                                                                entities_to_convert=entities_to_convert_ids)

            if entities_to_convert_ids is None or len(entities_to_convert_ids) == 0:
                continue
            entities_to_convert = [dataset.entity_id_2_name[x] for x in entities_to_convert_ids]

            rule_facts_with_relevance = []
            for cur_rule_with_relevance in rule_samples_with_relevance:
                cur_rule_samples, cur_relevance = cur_rule_with_relevance

                cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
                cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
                rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))

            print(";".join(fact))
            print(", ".join(entities_to_convert))
            print(", ".join(rule_facts_with_relevance))
            print()
            output_lines.append(";".join(fact) + "\n")
            output_lines.append(",".join(entities_to_convert) + "\n")
            output_lines.append(",".join(rule_facts_with_relevance) + "\n")
            output_lines.append("\n")

        elif args.mode == "necessary":
            try:
                rule_samples_with_relevance, complexity = kelpie.explain_necessary(sample_to_explain=sample_to_explain,
                                                                perspective="head",
                                                                num_promising_samples=args.prefilter_threshold)
                print(complexity)
                # remove debugging values of early exit #TODO: remove debuggin lines
                statistics["l.599>xsi"] += complexity.pop("l.599>xsi")
                statistics["l.552>xsi"] += complexity.pop("l.552>xsi")
                statistics["l.526>xsi"] += complexity.pop("l.526>xsi")
                statistics["l.484>xsi"] += complexity.pop("l.484>xsi")
                if "no partial set" in complexity.keys(): 
                    statistics["no partial set"] += complexity.pop("no partial set")

                statistics["complexity_sum"] += sum(complexity.values())
                statistics["facts"] += 1
            except EmptyResultException as e:
                print(fact)
                print(e)
                statistics['empty'].append(sample_to_explain)
                continue
        
            rule_facts_with_relevance = []
            for cur_rule_with_relevance in rule_samples_with_relevance:
                cur_rule_samples, cur_relevance = cur_rule_with_relevance

                cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
                cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
                rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))

            if verbose:
                print(";".join(fact))
                print(", ".join(rule_facts_with_relevance))
                print()
            output_lines.append(";".join(fact) + "\n")
            output_lines.append("|".join(rule_facts_with_relevance) + "\n")
            output_lines.append("\n")

        if 0 in complexity: 
            print(f"complexity of approach: {sum(complexity.values())} retrainings, {complexity[0]} for preliminary relevance")
        elif bool(complexity): 
            print(f"complexity of approach: {sum(complexity.values())} retrainings")

    end_time = time.time()
    duration = end_time - start_time
    #print("Required time: " + str(duration) + " seconds")
    statistics["time"] = duration
    statistics["avg_time"] = round( duration / statistics["facts"], 1)
    statistics["avg_complexity"] = round( statistics["complexity_sum"] / statistics["facts"], 1)
    statistics.pop("complexity_sum")
    statistics.pop("time")

    print(filename_9_output)
    with open(filename_9_output, "w") as output:
        output.writelines(output_lines)

    filename_original_output = os.path.join(DATA_PATH, dataset_name, "output.csv")
    with open(filename_original_output, "w") as original_output:
        original_output.writelines(output_lines)

    output_lines.append(str(statistics))
    print(filename_8_explain)
    with open(filename_8_explain, "a") as output:
        output.writelines(output_lines)

    filename_explain_output = os.path.join(OUTPUT_PATH, f"{dataset_name}_statistics.csv")
    print(filename_explain_output)
    statistics["algorithm"] = builder
    statistics["input_facts"] = facts_to_explain_path
    if not os.path.isfile(filename_explain_output):
        with open(filename_explain_output, "w") as output:
            output.writelines(";".join([str(val) for val in statistics.keys()])+"\n")
    with open(filename_explain_output, "a") as output:
        output.writelines(";".join([str(val) for val in statistics.values()])+"\n")

    return statistics

if __name__ == "__main__":
    main()