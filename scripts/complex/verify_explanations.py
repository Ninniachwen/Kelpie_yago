import sys
import os
import argparse
import copy
import random
import numpy
import pandas as pd
import torch

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from config import OUTPUT_PATH
from dataset import ALL_DATASET_NAMES, Dataset, MANY_TO_ONE, ONE_TO_ONE
from link_prediction.models.complex import ComplEx
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, REGULARIZER_WEIGHT, EPOCHS, \
    BATCH_SIZE, REGULARIZER_NAME
from utils import generate_output_file_prefix, seperate_rule_triples



def main():
    datasets = ALL_DATASET_NAMES

    parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

    parser.add_argument('--dataset',
                        choices=datasets,
                        help="Dataset in {}".format(datasets),
                        required=True)

    parser.add_argument('--model_path',
                        help="Path to the model to explain the predictions of",
                        required=True)

    optimizers = ['Adagrad', 'Adam', 'SGD']
    parser.add_argument('--optimizer',
                        choices=optimizers,
                        default='Adagrad',
                        help="Optimizer in {} to use in post-training".format(optimizers))

    parser.add_argument('--batch_size',
                        default=100,
                        type=int,
                        help="Batch size to use in post-training")

    parser.add_argument('--max_epochs',
                        default=200,
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
                        default=0,
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

    parser.add_argument("--mode",
                        type=str,
                        default="sufficient",
                        choices=["sufficient", "necessary"],
                        help="The explanation mode")

    parser.add_argument("--facts_to_verify_path",
                        type=str,
                        default="data/FR_Reduced_2K/output.csv",
                        help="The file with facts which shall be explained")

    args = parser.parse_args()

    # deterministic!
    seed = 42
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    kernel = ""
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        kernel = "cuda"
    else:
        torch.set_rng_state(torch.get_rng_state())
        kernel = "cpu"

    statistics = {"samples": 0, "original_direct_score": 0,    "new_direct_score": 0,    "original_tail_rank": 0,    "new_tail_rank": 0, "rank_change": 0,"score_change": 0}

    #############  LOAD DATASET
    # load the dataset and its training samples
    print("Loading dataset %s..." % args.dataset)
    dataset = Dataset(name=args.dataset, separator="\t", load=True)

    # get the ids of the elements of the fact to explain and the perspective entity
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

    output_prefix, timestamp = generate_output_file_prefix()
    filename_logs = output_prefix + "_11_verify_py.txt"
    print(filename_logs)
    filename_end_to_end_output = output_prefix + f"_12_{args.dataset}_output_end_to_end.txt"
    print(filename_end_to_end_output)

    with open(filename_logs, "w") as execution_log:
        execution_log.write(f"verify_explanations.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}, output: {filename_end_to_end_output}\n")

    with open(args.facts_to_verify_path, "r") as input_file:
        input_lines = input_file.readlines()

    original_model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: ComplEx
    if torch.cuda.is_available():
        original_model.to('cuda')
    original_model.load_state_dict(torch.load(args.model_path))
    original_model.eval()

    facts_to_explain = []
    samples_to_explain = []
    perspective = "head"    # for all samples the perspective was head for simplicity
    sample_to_explain_2_best_rule = {}

    if args.mode == "sufficient":

        sample_to_explain_2_entities_to_convert = {}

        i = 0
        while i <= len(input_lines)-4:
            fact_line = input_lines[i]
            similar_entities_line = input_lines[i+1]
            rules_line = input_lines[i+2]
            empty_line = input_lines[i+3]

            # sample to explain
            fact = tuple(fact_line.strip().split(";"))
            facts_to_explain.append(fact)
            sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
            samples_to_explain.append(sample)

            # similar entities
            similar_entities_names = similar_entities_line.strip().split(",")
            similar_entities = [dataset.entity_name_2_id[x] for x in similar_entities_names]
            sample_to_explain_2_entities_to_convert[sample] = similar_entities

            # rules
            rules_with_relevance = []

            rule_relevance_inputs = rules_line.strip().split(",")
            best_rule, best_rule_relevance_str = rule_relevance_inputs[0].split(":")
            best_rule_bits = best_rule.split(";")

            best_rule_facts = []
            j = 0
            while j < len(best_rule_bits):
                cur_head_name = best_rule_bits[j]
                cur_rel_name = best_rule_bits[j+1]
                cur_tail_name = best_rule_bits[j+2]

                best_rule_facts.append((cur_head_name, cur_rel_name, cur_tail_name))
                j+=3

            best_rule_samples = [dataset.fact_to_sample(x) for x in best_rule_facts]
            relevance = float(best_rule_relevance_str)
            rules_with_relevance.append((best_rule_samples, relevance))

            sample_to_explain_2_best_rule[sample] = best_rule_samples
            i += 4

        samples_to_add = []         # the samples to add to the training set before retraining
        samples_to_convert = []     # the samples that, after retraining, should have changed their predictions

        # for each sample to explain, get the corresponding similar entities and the most relevant sample in addition.
        # For each of those similar entities create
        #   - a version of the sample to explain that features the similar entity instead of the entity to explain
        #   - a version of the most relevant sample to add that features the similar entity instead of the entity to explain

        sample_to_convert_2_original_sample_to_explain = {}
        samples_to_convert_2_added_samples = {}
        for sample_to_explain in samples_to_explain:

            entity_to_explain = sample_to_explain[0] if perspective == "head" else sample_to_explain[2]

            cur_entities_to_convert = sample_to_explain_2_entities_to_convert[sample_to_explain]

            cur_best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]

            for cur_entity_to_convert in cur_entities_to_convert:
                cur_sample_to_convert = Dataset.replace_entity_in_sample(sample=sample_to_explain,
                                                                        old_entity=entity_to_explain,
                                                                        new_entity=cur_entity_to_convert,
                                                                        as_numpy=False)
                cur_samples_to_add = Dataset.replace_entity_in_samples(samples=cur_best_rule_samples,
                                                                    old_entity=entity_to_explain,
                                                                    new_entity=cur_entity_to_convert,
                                                                    as_numpy=False)

                samples_to_convert.append(cur_sample_to_convert)
                samples_to_convert_2_added_samples[cur_sample_to_convert] = cur_samples_to_add

                for cur_sample_to_add in cur_samples_to_add:
                    samples_to_add.append(cur_sample_to_add)

                sample_to_convert_2_original_sample_to_explain[tuple(cur_sample_to_convert)] = sample_to_explain

        new_dataset = copy.deepcopy(dataset)


        # if any of the samples_to_add overlaps contradicts any pre-existing facts
        # (e.g. adding "<Obama, born_in, Paris>" when the dataset already contains "<Obama, born_in, Honolulu>")
        # we need to remove such pre-eisting facts before adding the new samples_to_add
        print("Adding samples: ")
        for (head, relation, tail) in samples_to_add:
            print("\t" + dataset.printable_sample((head, relation, tail)))
            if new_dataset.relation_2_type[relation] in [MANY_TO_ONE, ONE_TO_ONE]:
                for pre_existing_tail in new_dataset.to_filter[(head, relation)]:
                    new_dataset.remove_training_sample(numpy.array((head, relation, pre_existing_tail)))

        # append the samples_to_add to training samples of new_dataset
        # (and also update new_dataset.to_filter accordingly)
        new_dataset.add_training_samples(numpy.array(samples_to_add))

        # obtain tail ranks and scores of the original model for that all_samples_to_convert
        original_scores, original_ranks, original_predictions = original_model.predict_samples(numpy.array(samples_to_convert))

        new_model = ComplEx(dataset=new_dataset, hyperparameters=hyperparameters, init_random=True)  # type: ComplEx
        new_optimizer = MultiClassNLLOptimizer(model=new_model, hyperparameters=hyperparameters)
        new_optimizer.train(train_samples=new_dataset.train_samples)
        new_model.eval()

        new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_convert))

        for i in range(len(samples_to_convert)):
            cur_sample = samples_to_convert[i]
            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]],
                                dataset.relation_id_2_name[cur_sample[1]],
                                dataset.entity_id_2_name[cur_sample[2]]]) + ">")
            print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_direct_score))
            print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank))
            print()

        output_lines = []
        for i in range(len(samples_to_convert)):
            cur_sample_to_convert = samples_to_convert[i]
            cur_added_samples = samples_to_add[i]
            original_sample_to_explain = sample_to_convert_2_original_sample_to_explain[tuple(cur_sample_to_convert)]

            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            # original_head, original_relation, original_tail,
            # fact_to_convert_head, fact_to_convert_rel, fact_to_convert_tail,

            # original_direct_score, new_direct_score,
            #  original_tail_rank, new_tail_rank

            a = ";".join(dataset.sample_to_fact(original_sample_to_explain))
            b = ";".join(dataset.sample_to_fact(cur_sample_to_convert))

            c = []
            samples_to_add_to_this_entity = samples_to_convert_2_added_samples[cur_sample_to_convert]
            for x in range(4):
                if x < len(samples_to_add_to_this_entity):
                    c.append(";".join(dataset.sample_to_fact(samples_to_add_to_this_entity[x])))
                else:
                    c.append(";;")

            c = ";".join(c)
            d = str(original_direct_score) + ";" + str(new_direct_score)
            e = str(original_tail_rank) + ";" + str(new_tail_rank)
            output_lines.append(";".join([a, b, c, d, e]) + "\n")

        with open("output_end_to_end.csv", "w") as outfile:
            outfile.writelines(output_lines)



    elif args.mode == "necessary":
        i = 0
        while i <= len(input_lines) -3:
            fact_line = input_lines[i]
            rules_line = input_lines[i + 1]
            empty_line = input_lines[i + 2]

            # if unexplained, skip
            if len(rules_line) < 3:
                print(f"skipping: '{fact_line}' because rules_line too small: '{rules_line}'")
                i += 3
                continue

            # sample to explain
            fact = tuple(fact_line.strip().split(";"))
            facts_to_explain.append(fact)
            sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
            samples_to_explain.append(sample)

            # rules
            rules_with_relevance = []

            rule_relevance_inputs = rules_line.strip().split("|")
            best_rule_relevance_str, best_rule = rule_relevance_inputs[0].split(":")
            best_rule_list = best_rule.strip().split(";")
            best_triples = [rule.strip().split(",") for rule in best_rule_list]
            
            length = [len(item) for item in best_triples]
            if max(length) > 3:
                best_triples = seperate_rule_triples(best_triples)

            best_rule_samples = [dataset.fact_to_sample(x) for x in best_triples]
            relevance = float(best_rule_relevance_str)
            rules_with_relevance.append((best_rule_samples, relevance))

            sample_to_explain_2_best_rule[sample] = best_rule_samples
            i += 3

        samples_to_remove = []  # the samples to remove from the training set before retraining

        for sample_to_explain in samples_to_explain:
            best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]
            samples_to_remove += best_rule_samples

        new_dataset = copy.deepcopy(dataset)

        print("Removing samples: ")
        for (head, relation, tail) in samples_to_remove:
            print("\t" + dataset.printable_sample((head, relation, tail)))

        # remove the samples_to_remove from training samples of new_dataset (and update new_dataset.to_filter accordingly)
        new_dataset.remove_training_samples(numpy.array(samples_to_remove))

        # obtain tail ranks and scores of the original model for all samples_to_explain
        original_scores, original_ranks, original_predictions = original_model.predict_samples(numpy.array(samples_to_explain))

        ######

        new_model = ComplEx(dataset=new_dataset, hyperparameters=hyperparameters, init_random=True)  # type: ComplEx
        new_optimizer = MultiClassNLLOptimizer(model=new_model, hyperparameters=hyperparameters)
        new_optimizer.train(train_samples=new_dataset.train_samples)
        new_model.eval()

        new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_explain))

        for i in range(len(samples_to_explain)):
            cur_sample = samples_to_explain[i]
            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]],
                                dataset.relation_id_2_name[cur_sample[1]],
                                dataset.entity_id_2_name[cur_sample[2]]]) + ">")
            print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_direct_score))
            print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank))
            print()

        output_lines = []
        for i in range(len(samples_to_explain)):
            cur_sample_to_explain = samples_to_explain[i]

            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            # original_head, original_relation, original_tail,
            # fact_to_convert_head, fact_to_convert_rel, fact_to_convert_tail,

            # original_direct_score, new_direct_score,
            #  original_tail_rank, new_tail_rank

            a = ";".join(dataset.sample_to_fact(cur_sample_to_explain))

            b = []
            samples_to_remove_from_this_entity = sample_to_explain_2_best_rule[cur_sample_to_explain]
            for x in range(4):
                if x < len(samples_to_remove_from_this_entity):
                    b.append(";".join(dataset.sample_to_fact(samples_to_remove_from_this_entity[x])))
                else:
                    b.append(";;")

            b = ";".join(b)
            c = str(original_direct_score) + ";" + str(new_direct_score)
            d = str(original_tail_rank) + ";" + str(new_tail_rank)
            output_lines.append(";".join([a, b, c, d]) + "\n")

            statistics["samples"] += 1
            statistics["original_direct_score"] += original_direct_score
            statistics["new_direct_score"] += new_direct_score
            statistics["original_tail_rank"] += original_tail_rank
            statistics["new_tail_rank"] += new_tail_rank
            statistics["rank_change"] += new_direct_score - original_direct_score
            statistics["score_change"] += new_tail_rank - original_tail_rank


        print(filename_end_to_end_output)
        with open(filename_end_to_end_output, "w") as outfile:
            outfile.writelines(output_lines)
    
    filename_statistics = os.path.join(OUTPUT_PATH, f"{dataset.name}_statistics.csv")
    print(filename_statistics)
    statistics["original_direct_score"] = statistics["original_direct_score"] / statistics["samples"]
    statistics["new_direct_score"] = statistics["new_direct_score"] / statistics["samples"]
    statistics["original_tail_rank"] = statistics["original_tail_rank"] / statistics["samples"]
    statistics["new_tail_rank"] = statistics["new_tail_rank"] / statistics["samples"]
    statistics["rank_change"] = statistics["rank_change"] / statistics["samples"]
    statistics["score_change"] = statistics["score_change"] / statistics["samples"]
    
    df = pd.read_csv(filename_statistics, sep=";", decimal=".")
    if 'rank_change' not in df:
        df["num_samples"] = ""
        df["original_direct_score"] = ""
        df["new_direct_score"] = ""
        df["original_tail_rank"] = ""
        df["new_tail_rank"] = ""
        df["rank_change"] = ""
        df["score_change"] = ""
    row = list(df.iloc[-1,:-7])
    row.extend(list(statistics.values()))
    new_df = df.iloc[:-1,:]
    new_df.loc[len(new_df)] = row
    new_df.to_csv(filename_statistics, sep=';', index=False)

if __name__ == "__main__":
    main()