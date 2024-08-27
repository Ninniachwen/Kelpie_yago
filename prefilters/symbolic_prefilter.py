from multiprocessing.pool import ThreadPool as Pool
import threading

from dataset import Dataset
from link_prediction.models.model import Model
from prefilters.prefilter import PreFilter
from ruleset import Ruleset
from config import MAX_PROCESSES

class SymbolicPreFilter(PreFilter):
    """
    The SymbolicPreFilter object is a PreFilter that relies on symbolic rules
    to extract the most promising samples for an explanation.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 ruleset: Ruleset,
                 verbose = True):
        """
        PostTrainingPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param ruleset: the rules belonging to this dataset and model
        :param verbose: extensive print output if True
        """
        super().__init__(model, dataset)

        self.max_path_length = 5
        self.entity_id_2_train_samples = {}     # dictionary of all entity ids (int) and entity names(str)
        self.threadLock = threading.Lock()
        self.counter = 0
        self.thread_pool = Pool(processes=MAX_PROCESSES)
        self.ruleset = ruleset
        self.verbose = verbose

        for (h, r, t) in dataset.train_samples:     # iterate over all triples in Graph

            # sort triples into mention groups { a : [(a,r1,b), (c,r2,a)], b: ... }
            if h in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[h].append((h, r, t))
            else:
                self.entity_id_2_train_samples[h] = [(h, r, t)]

            if t in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[t].append((h, r, t))
            else:
                self.entity_id_2_train_samples[t] = [(h, r, t)]

    def top_promising_samples_for(self,
                                  sample_to_explain:list[int, int, int],
                                  perspective:str,
                                  top_k=50) \
                                    -> tuple[list[list[int, int, int]], list[list[int, int, int]]]:

        """
        This method extracts the top k promising samples for interpreting the sample to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising samples featuring the head of the sample to explain
                                - if "tail", find the most promising samples featuring the tail of the sample to explain
        :param top_k: the number of top promising samples to extract.
        :return: the sorted list of the k most promising samples. Or an empty list. The samples are real triples as they appear in the graph, not the general rules.
        """
        self.counter = 0

        # split predicted triple into its components
        head_id, relation_id, tail_id = sample_to_explain

        if self.verbose:
            print(f"(symbolic prefilter) Extracting promising facts for <{head_id}, {relation_id}, {tail_id}>")
 
        # determine which entity was predicted and which is central entity
        start_entity_id, end_entity_id = (head_id, tail_id) if perspective == "head" else (tail_id, head_id)

        # if central entity not in training set, then prediction can not be explained
        if start_entity_id not in self.entity_id_2_train_samples:
            print(f"skipping explanation for triple '{self.dataset.printable_sample(sample_to_explain)}': '{self.dataset.entity_id_2_name[start_entity_id]}' not in training set")
            return []

        ## find all relevant triples conntected to entity using rules (symbolic learning)

        # identify rules which apply to this predictions relation
        nec_evidence, suf_evidence = self.ruleset.find_rules_and_evidence(relation_id, end_entity_id, start_entity_id, top_k) # include end_entity, to accomodate rules with constants
        # only keep rules which apply to this entity -> evidence in ego-graph

        if len(nec_evidence) == 0:
            if self.verbose: print(f"no rule applies to this entity: '{start_entity_id}'")
            return [], []

        return nec_evidence, suf_evidence
