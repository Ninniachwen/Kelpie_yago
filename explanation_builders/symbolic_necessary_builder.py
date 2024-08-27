import collections

from dataset import Dataset
from relevance_engines.post_training_engine import PostTrainingEngine
from link_prediction.models.model import Model
from explanation_builders.explanation_builder import NecessaryExplanationBuilder
from ruleset import RuleEvidence, Ruleset
from utils import flatten_list


class SymbolicNecessaryExplanationBuilder(NecessaryExplanationBuilder):
    """
    The StochasticNecessaryExplanationBuilder object guides the search for necessary rules with a probabilistic policy
    """

    def __init__(self, model: Model,
                 dataset: Dataset,
                 ruleset: Ruleset,
                 hyperparameters: dict,
                 sample_to_explain: tuple[int, int, int],
                 perspective: str,
                 output_prefix: str,
                 relevance_threshold: float = None,
                 max_explanation_length: int = -1,
                 verbose: bool = True):
        """
        StochasticSufficientExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param sample_to_explain: the predicted sample to explain
        :param perspective: the explanation perspective, either "head" or "tail"
        :param max_explanation_length: the maximum number of facts to include in the explanation to extract
        """

        super().__init__(model=model, dataset=dataset,
                         sample_to_explain=sample_to_explain, perspective=perspective,
                         max_explanation_length=max_explanation_length,
                         output_prefix=output_prefix)

        engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)
        
        # compute complexity -> number of rules checked for relevance
        self.verbose = verbose
        self.length_cap = 5 #TODO: what value best and make arg?

        self.ruleEvidence = RuleEvidence(ruleset=ruleset, dataset=dataset, engine=engine, perspective=perspective, sample_to_explain=sample_to_explain, output_prefix=output_prefix, relevance_threshold=relevance_threshold, verbose=verbose)

    def build_explanations(self,
                           rules_to_remove: list[list[list[int, int, int]]],
                           top_k: int = 10, algorithm="rules_kelpie"):
        
        self.ruleEvidence.set_rules_to_remove(rules_to_remove, top_k)
        
        if "reverse" in algorithm:
            all_rules_with_relevance, complexity = self.kelpie_rule_reverse_alg(rules_to_remove)
        elif "pca" in algorithm:
            all_rules_with_relevance, complexity = self.pca_rule_alg(rules_to_remove)
        elif "frequency" in algorithm:
            all_rules_with_relevance, complexity = self.frequency_rule_alg(rules_to_remove)
        elif "kelpie" in algorithm:
            all_rules_with_relevance, complexity = self.kelpie_rule_alg(rules_to_remove)
        else:
            raise Exception("algorithm type not recognized. Try 'rules_kelpie' of 'rules_heuristic_pca'")
        
        if len(all_rules_with_relevance) > top_k:
            raise Exception("top_k needs to be applied before this point!")
        return sorted(all_rules_with_relevance, key=lambda x: x[1], reverse=True), complexity # TODO: [:top_k] probably not necessary any more
    

    def kelpie_rule_alg(self, rules_to_remove):
        print(f"\n----- Triple Relevance: {len(self.ruleEvidence.relevant_triples_set)} unique triples -------")
        # get relevance for each triple, for later calculation of preliminary score
        sample_2_relevance = self.ruleEvidence.extract_sample_relevance(self.ruleEvidence.relevant_triples_set)
        
        
        print(f"\n----- Single Rule Relevance: {len(rules_to_remove)} different rules -------")
        # get relevance for all rules
        rule_2_preliminary_relevance = self.ruleEvidence.extract_rule_relevance()
        print(rule_2_preliminary_relevance)
        
        self.ruleEvidence.short_print_rule_relevance(rule_2_preliminary_relevance)

        # combined rule relevance (print in method)
        all_rules_with_relevance, complexity = self.ruleEvidence.extract_rule_relevance_combinatorial(self.length_cap)

        return all_rules_with_relevance, complexity
    

    def kelpie_rule_reverse_alg(self, rules_to_remove: list[list[list[int]]]):
        print(f"\n----- Full Set Relevance: {len(rules_to_remove)} rules -------")
        # get relevance for the full rule set
        full_set_relevance = self.ruleEvidence.extract_set_relevance(rules_to_remove)
        
        # if only one rule applicable, onlyone relevance calculation is necessary
        if len(rules_to_remove) == 1:
            return [[rules_to_remove, full_set_relevance]], self.ruleEvidence.get_complexity()


        print(f"\n----- Partial Set Relevance: {len(rules_to_remove)} versions -------")
        # get relevance for ruleset 
        necessary_set = self.ruleEvidence.extract_partial_set_relevance(full_set_relevance)

        print(f"\n----- Necessary Set Relevance: {len(necessary_set)} rules -------")
        # get relevance for the full rule set
        nec_set_relevance = self.ruleEvidence.extract_set_relevance(necessary_set)

        return [(tuple(necessary_set), nec_set_relevance)], self.ruleEvidence.get_complexity()
    

    def pca_rule_alg(self, rules_to_remove):
        print(f"\n----- Single Rule PCA Confidence: {len(rules_to_remove)} different rules -------")
        # get relevance for all rules
        rule_2_confidence = self.ruleEvidence.extract_rule_confidence(mode="necessary", rules=rules_to_remove)

        # sort by relevance and store as list
        rule_confidence_list = sorted(rule_2_confidence.items(), key=lambda x: x[1], reverse=True)
        self.ruleEvidence.short_print_rule_relevance(rule_confidence_list)

        print(f"\n----- Combined Rule Relevance: max length {len(rules_to_remove)} Rules -------")
        # get relevance incrementally
        all_rules_with_relevance, complexity = \
            self.ruleEvidence.extract_rule_relevance_incrementally(rule_confidence_list)
        #self.ruleEvidence.short_print_rule_relevance(all_rules_with_relevance)

        return all_rules_with_relevance, complexity


    def frequency_rule_alg(self, rules_to_remove):
        triples = flatten_list(rules_to_remove)
        print(f"\n----- Single triple frequency: {len(set(triples))} unique triples in rules -------")
        # get relevance for all rules
        triple_2_frequency = collections.Counter(triples)

        # sort by relevance and store as list
        triple_frequency_list = sorted(triple_2_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # pack triples in list, so later handling works
        for i in range(len(triple_frequency_list)):
            triple, value = triple_frequency_list[i]
            triple_frequency_list[i] = ([triple], value)

        self.ruleEvidence.short_print_rule_relevance(triple_frequency_list)

        print(f"\n----- Combined Triple Relevance: max length {len(triple_frequency_list)} Triples -------")
        # get relevance incrementally
        all_rules_with_relevance, complexity = \
            self.ruleEvidence.extract_rule_relevance_incrementally(triple_frequency_list)
        #self.ruleEvidence.short_print_rule_relevance(all_rules_with_relevance)

        return all_rules_with_relevance, complexity
    