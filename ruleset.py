import copy
import itertools
import operator
import os
import random
import re
import duckdb
import numpy
import pandas as pd
from rdflib import Graph, URIRef
import rdflib

from utils import flatten_list
from config import ROOT
from dataset import Dataset
from relevance_engines.post_training_engine import PostTrainingEngine

RULE_PATH = os.path.join(ROOT, "rules")
DEAFAULT_XSI_THRESHOLD = 5
IMPROVEMENT_THRESHOLD = 0.2
ABSURDLY_LOW_VALUE = -1e6  # absurdly low value for initialization

class Ruleset:

    def __init__(self,
                 dataset: Dataset,
                 rules_file1: str,
                 rules_file2: str = "",
                 pca_threshold:float = 0.7,
                 order_col = 'pca_confidence',
                 load: bool = True,
                 sep:str ='\t',
                 decimal:str = '.',
                 verbose: bool = True):
        """
            Ruleset constructor.
            This method will initialize the Ruleset and its structures.
            If parameter "load" is set to true, it will immediately read the dataset files and fill the data structures with the read data.

        Args:
            rules_file (str): _description_
            rules_file2 (str): _description_
            pca_threshold (float, optional): _description_. Defaults to 0.7.
            order_col (str, optional): _description_. Defaults to 'pca_confidence'.
            load (bool, optional): boolean flag; if True, the dataset files must be accessed and read immediately.. Defaults to True.
            sep (str, optional): the character that separates columns in the rules files. Defaults to ','.
            decimal (str, optional): _description_. Defaults to '.'.
            verbose (bool, optional): _description_. Defaults to True.

        Raises:
            Exception: _description_
        """        

        self.dataset = dataset
        self.order_col:str = order_col
        self.verbose:bool = verbose
        self.dataFrame:pd.DataFrame = None
        self.rules_path_1 = os.path.join(RULE_PATH, rules_file1)
        if rules_file2 != None:
            self.rules_path_2 = os.path.join(RULE_PATH, rules_file2)
        else:
            self.rules_path_2 = None
    
        self.sufficient_row_ids = {}
        self.necessary_row_ids = {}
        
        if load:
            if not os.path.isfile(self.rules_path_1):
                raise Exception("File %s does not exist" % self.rules_path_1)
            
            # Rule import        #TODO: cover different formats or use auto format detection?
            rules_df1 = pd.read_csv(self.rules_path_1, sep=sep, decimal=decimal, header=0, encoding='utf8')  # load rules
            if verbose: print(f"loading Rules...   {self.rules_path_1}")

            if self.rules_path_2 != None:
                if os.path.isfile(self.rules_path_2):
                    if verbose: print(f"joining editorial Rules...   {self.rules_path_2}")
                    rules_df2 = pd.read_csv(self.rules_path_2, sep=sep, decimal=decimal, header=0)  # load editorial rules
                    self.dataFrame = pd.concat([rules_df1,rules_df2], axis=0, ignore_index=True).fillna(0.99)
                else:
                    if verbose: print("rulefile 2 path not valid: {self.rules_path_2}. Skipping editorial rules file")
            else:
                if verbose: print("no editorial Rules")
                self.dataFrame = rules_df1

            self.dataFrame.columns = self.dataFrame.columns.str.replace(' ', '_')
            #self.rules_df = self.rules_df

            # filter and reduce rules_df 
            self.dataFrame = self.dataFrame.loc[self.dataFrame[self.order_col] > pca_threshold]
            self.dataFrame = self.dataFrame.loc[self.dataFrame[self.order_col] != 1]
            self.dataFrame = self.dataFrame[['body', 'head', self.order_col, 'functional_variable']]

            # different file format
            if "Rule" in self.dataFrame.columns:
                self.dataFrame[['body', 'head']] = self.dataFrame['Rule'].apply(lambda x: pd.Series(self._split_hornrule_into_head_body(x)))  # split rule into body and head triples
                self.dataFrame = self.dataFrame.drop('Rule', axis=1)   # remove original rule string

            self.dataFrame["head"] = self._remove_unwanted_characters(self.dataFrame["head"])
            self.dataFrame["body"] = self._remove_unwanted_characters(self.dataFrame["body"])

            self.dataFrame.reset_index(drop=True, inplace=True)

            
    @staticmethod
    def _remove_unwanted_characters(df_col):
        # same as in dataset.py line 163
        col1 = df_col.str.replace(",", "")
        col1 = col1.str.replace(":", "")
        col1 = col1.str.replace(";", "")
        col1 = col1.str.replace(".", "")
        return col1


    # https://duckdb.org/2021/05/14/sql-on-pandas.html   replaces # = sqldf(q1, locals())     #deprecated
    def find_rules_and_evidence(self, relation_id:str, target_id:str, start_entity_id:int, top_k:int)->  tuple[list[list[int, int, int]], list[list[int, int, int]]]:
        rules_df = self.dataFrame
        relation = self.dataset.relation_id_2_name[relation_id]
        target = self.dataset.entity_id_2_name[target_id]

        if rules_df['head'].str.contains(target).any():     # if target entity is contained in rule_df -> rules with constants
            q1 = f"""SELECT * FROM rules_df WHERE head LIKE '%{relation}%{target}%' OR head LIKE '%{relation}%?%' ORDER BY {self.order_col} DESC"""     # include rules with correct constant and variable as target
            #TODO: this only covers head predictions?
        else:
            q1 = f"""SELECT * FROM rules_df WHERE head LIKE '%{relation}%' ORDER BY {self.order_col} DESC"""

        relevant_rules = duckdb.query(q1).df()

        if self.verbose:
            print(q1)
            print("relevant_rules:")
            print(relevant_rules)
            print()
        
        if relevant_rules.empty:
            raise EmptyResultException(f"no rules applicable to the relation '{relation}'")
        
        self.local_graph = self._load_graph(central_entity=start_entity_id, radius=self._longest_rule_length(relevant_rules))

        return self._find_evidence_via_ids(relevant_rules, start_entity_id, top_k)
    
    @staticmethod
    def _longest_rule_length(relevant_rules:pd.DataFrame):
        entry_count = relevant_rules['body'].str.count('  ').max() + 1
        # 2 spaces = 3 entries = 1 triple
        # 5 spaces = 6 entries = 2 triples

        if (entry_count % 3) != 0:   #if its not a full integer, there must be a mistake
            raise Exception(f"counted {max} entries, but number of entries must be divisible by 3")
        
        return int(entry_count/3)
        

    def _load_graph(self, central_entity:int=None, radius:int=None) -> Graph:
        g1 = Graph()
        for line in self.dataset.train_samples:
            #triple = [(prefix + t) for t in triple]
            triple = (URIRef(str(t)) for t in line) # we have to wrap them in URIRef
            g1.add(triple)                    # and add to the graph
            # Successfully parsed this line
        
        #if (central_entity != None) and (type(range)==int):
        #    query = "PREFIX : <> SELECT DISTINCT ?s ?p ?o WHERE {"
        #    for i in range(radius):
        #        query = query + "{ " + central_entity + "?p1 ?o1"
        #    query = f"{query} WHERE {{{rule_string}}}"
        #    # if self.verbose: print(query)
            
        #    qres = graph.query(query)

        return g1
    
    
    def _find_evidence_via_ids(self, relevant_rule_df:pd.DataFrame, entity_id:int, top_k:int)  \
        -> tuple[list[list[int, int, int]], list[list[int, int, int]]]:
        """_summary_

        Args:
            rule_df (_type_): _description_
            entity_id (_type_): _description_
            top_k (_type_): _description_
            threshold (int, optional): _description_. Defaults to 0.

        Returns:
            list[tuple[list[str]]]: a list of rule tuples, each triple in a rule is represented as a list of strings. For each rule #TODO more than one result per rule...
        """

        self.nec_result = [] # sufficient: we need each instanciation of a rule in seperate list, to add single instaces/chains to convert
        self.suf_result = [] # necessary we need a single list of all instanciations of a rule, to remove at once to reduce score

        entity_name = self.dataset.entity_id_2_name[entity_id]
        self.num_samples_found = 0

        for idx, item in relevant_rule_df.iterrows():    # for each applicable rule
            if self.num_samples_found > top_k:
                break

            fun_var:str = item['functional_variable']
            body:str = item['body']
            head:str = item['head']

            # Replace functional variable in body with central entity_id
            body = body.replace(fun_var, entity_name)   # TODO quicker if not name but id, but it complicates line 161ff ... "replace relatios & entities by id's"


            try:
                variables, rule_string, triples = \
                    self._extract_vars_and_rules(body)
            except RuleNotApplicableException as e:
                # if this rule can not be properly transformed (str -> int) then skip this rule
                print(e)
                continue
            
            try:
                qres:list[rdflib.query.ResultRow] = self._query_graph_for_evidence(variables, rule_string)
                if self.verbose: print(f"Rule: {body} -> {head}")
            except EmptyResultException:
                continue

            suf, nec = self._extract_sets(qres, triples)

            # store row id for each suf & nec set, to later identify eg. pca confidence in self.dataFrame
            row_id = self.extract_row_id(item['body'], item['head'])

            self.sufficient_row_ids[tuple(suf)] = row_id
            self.necessary_row_ids[tuple(nec)] = row_id


        if self.verbose: print("suf: ", self.suf_result)
        if self.verbose: print("nec: ", self.nec_result)


        if len(self.suf_result) == 0:
            raise EmptyResultException("no rule applies to this entity, necessary and sufficient sets are empty")

        return self.nec_result, self.suf_result
    
    def extract_row_id(self, body, head):
        row_id = self.dataFrame.index[
            ((self.dataFrame['body'] == body) & 
             (self.dataFrame['head'] == head))
             ].tolist()
        
        if len(row_id) < 1:
            raise Exception(f"no rule matches this body and head:\nlooking for: {body} -> {head}\ndf: {self.dataFrame}")
        elif len(row_id) > 1:
            raise Exception(f"more than one rule matches this body and head:\nlooking for: {body} -> {head}\nrows: {row_id}: {self.dataFrame.iloc[row_id]}")
        
        return row_id[0]
            
    def _extract_vars_and_rules(self, body:str)-> tuple[list[str], list[str], list[list[str]]]:
        
        # identify variables like ?b for SPARQL query
        var_pattern = re.compile(r'[?]\w')
        variables:list[str] = var_pattern.findall(body)
        variables:set[str] = set(variables)

        # Split the rule body into individual words
        words = body.split()
        
        # Split the word-list of the rule into triples
        split_len = 3
        i = 0
        rule_string = ""
        triples = []

        error_in_triple = False
        while i <= (len(words)-split_len):   # for each triple in rule
            # extract each triple
            triple = words[i:(i+split_len)]

            # replace relatios & entities by id's
            for j, t in enumerate(triple):
                if re.match(var_pattern, t) == None:    # if its not a variable
                    if j==0 or j==2:    # for entities
                        try:
                            triple[j] = f":{self.dataset.entity_name_2_id[t.lower()]}"
                        except Exception as e:
                            print(f"entity '{t.lower()}' not in dict of entities: {e}. check if rules file matches dataset")
                            error_in_triple = True
                    elif j==1:  # for relations
                        try:
                            triple[j] = f":{self.dataset.relation_name_2_id[t.lower()]}"
                        except Exception as e:
                            print(f"relation '{t.lower()}' not in dict of relations: {e}. check if rules file matches dataset")
                            error_in_triple = True

            # create triple representation (and remove : ) for result
            triples.append([tri.replace(":", "") for tri in triple])

            # Join the parts into one string of rdf triples for query
            triple_string = ' '.join(triple)
            rule_string += triple_string + ". "
            i = i + split_len

            if error_in_triple:
                raise RuleNotApplicableException(f"error in rule {triple}, rule skipped")

        return variables, rule_string, triples

    def _query_graph_for_evidence(self, variables, rule_string)-> list[rdflib.query.ResultRow]:

        #TODO: only load partial graph?
        #TODO: store graph?
        graph = self.local_graph

        # if no variables, ask query
        if len(variables) == 0:
            query = f"PREFIX : <> ASK  {{{rule_string}}}"
            #if self.verbose: print(query)

            qres = graph.query(query)
            if qres.askAnswer == True:
                qres = qres
            # if query result is empty, continue search
            elif qres.askAnswer == False:
                raise EmptyResultException("no results in ask query")
            else:
                print("error reading qres boolean. ASK Query might have failed")

        # construct query from all variables and parts of rule body
        else:
            query = f"PREFIX : <> SELECT DISTINCT "
            for var in variables:
                query = query + var + " "
            query = f"{query} WHERE {{{rule_string}}}"
            # if self.verbose: print(query)
            
            qres = graph.query(query)
            
            # if query result is empty, continue search
            if len(qres) == 0:
                raise EmptyResultException("no results in variable query")
            else:
                self.num_samples_found += 1
        return qres
    
    def _extract_sets(self, qres, rule_triples)-> tuple[list, list]:
        nec = []
        suf = []

        # convert triples with variables into triples from Graph
        #lq = len(qres)                          #TODO: remove after debugging
        for items in qres:  # for each row in result = for each instanciation of rule
            instance = copy.deepcopy(rule_triples)
            #if lq > 1:                #TODO: remove after debugging
            #    print("MORE THAN 1 INSTANCE PER RULE")#TODO: remove after debugging
            new_triples = []
            for i, triple in enumerate(instance):    # check all triples
                if type(items) == bool:
                    items = [""]
                    var = [""]
                else:
                    var = iter(items.labels)
                for item, v in zip(items, var):   # for each variable in query
                    v = f"?{v}"
                    for j, t in enumerate(triple):
                        t_new = t
                        if type(t) == str:
                            t_new = t.replace(v, item) # replace variable with found entity
                            try:    # cast to int if no var
                                t_new = int(t_new)
                            except:
                                pass
                        instance[i][j] = t_new
                new_triples.append(tuple(triple))
            self.suf_result.append(tuple(new_triples))  # suf grouped by instanciation #TODO update tuple type
            suf.append(tuple(new_triples)) #suf set return (for storing with row id)
            nec += new_triples
        self.nec_result.append(tuple(nec)) # nec grouped by rule
        return suf, nec
    

    def _split_hornrule_into_head_body(self, rule:str)->tuple[list[str], list[str]]:
        # Split the rule into parts separated by '=>'
        left, right = rule.split('=>')
        body_triples = self._extract_statements(left)
        head_triples = self._extract_statements(right)
        
        return body_triples, head_triples
    
        
    def _extract_statements(self, triple_string:str)->list[str]:
        triple_string_clean = re.sub(r"\s+", ' ', triple_string)
        # Extract statements from longer string
        string_list = re.findall(r'[^\s]+ [^\s]+ [^\s]+', triple_string_clean)
        return string_list


    @staticmethod
    def extract_relations(rules:list[list[tuple[int, int, int]]]|list[tuple[int, int, int]]|tuple[int, int, int], dataset) -> list[str]|list[list[str]]:
        if (type(rules) == tuple) & (type(rules[0]) == int):    # single triple
            return dataset.sample_to_fact(rules)
        
        relation_list = []
        for i, group in enumerate(rules):
            if (len(group) == 3) & (type(group[0]) == int):     # group:tuple[int, int, int]
                sample = dataset.sample_to_fact(group)          # sample:tuple[str, str, str]
                relation_list.append(sample[1])                 # only append predicate
            elif (len(group[0]) == 3) & (type(group[0][0]) == int):
                relation_list.append([])
                for tup in group:               # tuple:tuple[int, int, int]
                    sample = dataset.sample_to_fact(tup)      # sample:tuple[str, str, str]
                    relation_list[i].append(sample[1])          # only append predicate
            else:
                relation_list.append(f"malformed group '{group}'")
        return relation_list


    @staticmethod
    def replace_entity_in_samples(samples:list[list[int, int, int]], old_entity: int, new_entity:int, as_numpy=True):
        result = []
        for (h, r, t) in samples:
            if h == old_entity:
                h = new_entity
            if t == old_entity:
                t = new_entity
            result.append((h, r, t))

        return numpy.array(result) if as_numpy else result
    

    def printable_nple(self, nple: list):
        return" + ".join([self.printable_sample(sample) for sample in nple])

    def printable_sample(self, sample: tuple[int, int, int]):
        return "<" + ", ".join(self.dataset.sample_to_fact(sample)) + ">"

class RuleEvidence:
    def __init__(self, ruleset: Ruleset,
                 dataset: Dataset,
                 engine:PostTrainingEngine,
                 perspective: str,
                 sample_to_explain:tuple[int, int, int],
                 output_prefix:str,
                 relevance_threshold: float = None,
                 verbose:bool = True):
        
        self.verbose = verbose
        self.ruleset:Ruleset = ruleset
        self.dataset:Dataset = dataset
        self.engine = engine
        self.perspective = perspective
        self.sample_to_explain = sample_to_explain
        self.output_prefix = output_prefix

        self.xsi = relevance_threshold if relevance_threshold is not None else DEAFAULT_XSI_THRESHOLD
        self.window_size = 10
        self.improvement_threshhold = IMPROVEMENT_THRESHOLD

        self.complexity:dict = {"l.599>xsi": [], "l.552>xsi" : [], "l.526>xsi" : [], "l.484>xsi" : [], "no partial set":[]} 
        self.sample_2_relevance:dict = {}
        self.prelim_rule_score:dict = {}


    ### getter - setter ###


    def set_rules_to_remove(self, rules_to_remove: list[list[list[int, int, int]]], top_k)->None:
        self.rules_to_remove = rules_to_remove
        self.relevant_triples_list = flatten_list(rules_to_remove) # turn list of lists into list of triples, with duplicates
        self.relevant_triples_set = tuple(self.relevant_triples_list) # turn list of triples into tuple of triples, without duplicates
        self.top_k = top_k
    
    def get_complexity(self):
        return self.complexity


    ### ---- ###


    def extract_sample_relevance(self, samples_to_remove: tuple[tuple[int, int, int]]):

        sample_2_preliminary_relevance = {}

        # all triples are tested for preliminary score
        for i, sample_to_remove in enumerate(samples_to_remove):
            if self.verbose: print("\nComputing preliminary relevance for triple " + str(i+1) + " on " + str(
                len(samples_to_remove)) + ": " + self.dataset.printable_sample(sample_to_remove))
            triple_relevance = self._compute_relevance_for_rule(([sample_to_remove]), length=0)
            sample_2_preliminary_relevance[sample_to_remove] = triple_relevance
            if self.verbose: print("\tObtained relevance: " + str(triple_relevance))
        self.sample_2_relevance = sample_2_preliminary_relevance
        return sample_2_preliminary_relevance
    
    
    def extract_set_relevance(self, rule_set:list[tuple[int, int, int]]|list[list[tuple[int, int, int]]])->float:
        if type(rule_set[0]) == int:
            raise Exception("should be list of triples")
        if type(rule_set[0][0]) != int:
            rule_set = flatten_list(rule_set)
        relevance = self._compute_relevance_for_rule(rule_set, len(rule_set))
        if self.verbose: print(f"{relevance}: {rule_set}")
        return relevance
    
    
    def extract_partial_set_relevance(self, full_set_relevance:float)->list:
        full_set = set(flatten_list(self.rules_to_remove))
        necessary_set = []
        leng = len(self.rules_to_remove) - 1
        confidence_dict = self.extract_rule_confidence(mode="necessary", rules=self.rules_to_remove)
        sorted_rule_confidence = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)

        # add single rules back to graph, if relevance of removed set decreases, this rule is important & added to necessary set
        unimportant = False
        for rule_triples, rule_conficendce in sorted_rule_confidence:
            partial_set = list(copy.deepcopy(full_set))
            for triple in rule_triples:
                partial_set.remove(triple)
            partial_relevance = self._compute_relevance_for_rule(partial_set, leng)
            #diff = full_set_relevance - partial_relevance   
            #if (diff) > self.improvement_threshhold:
            if full_set_relevance > partial_relevance:      # if partial relevance is lower, this rule is important
                necessary_set.append(list(rule_triples))
            print(f"pca: {round(rule_conficendce, 3)} | Relavance with {self.short_print_rule_relevance({rule_triples:partial_relevance}, inplace=False)}")
            """
                unimportant = False
            elif unimportant == True:   # end ealy, if two samples in a row are unimportant
                    break
            else:
                unimportant = True
            """

        if len(necessary_set) == 0:     # if none of the parts are important by themselves, then use all together   
            necessary_set = full_set
            self.complexity["no partial set"].append((f"{full_set_relevance}: {self.sample_to_explain}"))
            #raise Exception("no partial set found")     # TODO: remove after debugging
        return necessary_set
        

    def extract_rule_relevance(self):

        # combine preliminary triple score into score per rule
        all_rules_with_preliminary_scores = [
            (tuple(x), self._calc_preliminary_rule_score(x)) 
            for x in self.rules_to_remove]
        
        # sort rules by preliminary score
        all_rules_with_preliminary_scores = sorted(all_rules_with_preliminary_scores,
                                                            key=lambda x: x[1], reverse=True)

        rule_2_relevance_dict = {}

        terminate = False
        best_relevance_so_far = ABSURDLY_LOW_VALUE  # initialize with an absurdly low value

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        while i < len(all_rules_with_preliminary_scores) and not terminate:

            current_rule, current_preliminary_score = all_rules_with_preliminary_scores[i]

            # compute relevance for all single rules
            if self.verbose: print("\n\tComputing relevance for rule: " + self.dataset.printable_nple(current_rule))
            current_rule_relevance = self._compute_relevance_for_rule(current_rule, length=1)
            rule_2_relevance_dict[current_rule] = current_rule_relevance
            if self.verbose: print("\tObtained relevance: " + str(current_rule_relevance))

            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = current_rule_relevance

            # early termination if relevance is very high
            if current_rule_relevance > self.xsi:
                i += 1
                self.complexity["l.484>xsi"].append((self.sample_to_explain, current_rule_relevance))
                break

            # else, if the current relevance value is an improvement over the best relevance value seen so far, continue
            elif current_rule_relevance >= best_relevance_so_far:
                best_relevance_so_far = current_rule_relevance
                i += 1
                continue

            # else, if the window has not been filled yet, continue
            elif i < self.window_size:
                i += 1
                continue

            # else, use the average of the relevances in the window to assess the termination condition
            else:
                cur_avg_window_relevance = self._average(sliding_window)
                terminate_threshold = cur_avg_window_relevance / best_relevance_so_far
                random_value = random.random()
                terminate = random_value > terminate_threshold  # termination condition

                print("\n\tCurrent relevance " + str(current_rule_relevance))
                print("\tCurrent averaged window relevance " + str(cur_avg_window_relevance))
                print("\tMax relevance seen so far " + str(best_relevance_so_far))
                print("\tTerminate threshold:" + str(terminate_threshold))
                print("\tRandom value:" + str(random_value))
                print("\tTerminate:" + str(terminate))
                i += 1

        # sort rules by importance 
        all_rules_with_relevance_list = sorted(rule_2_relevance_dict.items(), key=lambda x: x[1], reverse=True)
        self.all_rules_with_relevance = all_rules_with_relevance_list
        self.rule_2_relevance = rule_2_relevance_dict
        return all_rules_with_relevance_list
    

    def extract_rule_relevance_combinatorial(self, length_cap:int) -> list[tuple[tuple,float]]:
        
        best_rule, best_rule_relevance = self.all_rules_with_relevance[0]
        if best_rule_relevance > self.xsi:
            print(f"early exit: rules with length 1")
            print(f"best_rule_relevance ({best_rule_relevance}) > self.xsi ({self.xsi})")
            self.complexity["l.526>xsi"].append((self.sample_to_explain, best_rule_relevance)) 
            
            return self.all_rules_with_relevance, self.complexity
        
        rules_number = len(self.rules_to_remove)
        cur_rule_combinations = 2
        print(f"\n----- Combined Rule Relevance: {2**rules_number-1-rules_number} possible combinations -------")

        # stop if you have too few samples (e.g. if you have only 2 samples, you can not extract rules of length 3)
        # or if you get to the length cap
        while cur_rule_combinations <= rules_number and cur_rule_combinations <= length_cap:
            rule_2_relevance = self._extract_rule_combination_relevance(
                length=cur_rule_combinations)
            current_rules_with_relevance = sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)

            self.all_rules_with_relevance += current_rules_with_relevance
            self.all_rules_with_relevance.sort(key=operator.itemgetter(1), reverse=True)
            self.all_rules_with_relevance = self.all_rules_with_relevance[:self.top_k]
            current_best_rule, current_best_rule_relevance = self.all_rules_with_relevance[0]

            if current_best_rule_relevance > best_rule_relevance:
                best_rule, best_rule_relevance = current_best_rule, current_best_rule_relevance
            # else:
            #   break       if searching for additional rules does not seem promising, you should exit now

            if best_rule_relevance > self.xsi:
                if self.verbose: print(f"early exit: cur_rule_length ({cur_rule_combinations}) \
                      \nbest_rule_relevance ({best_rule_relevance}) > self.xsi ({self.xsi})")
                self.complexity["l.552>xsi"].append((self.sample_to_explain, best_rule_relevance)) 
                break

            #if len(self.all_rules_with_relevance) > self.top_k:
            #    if self.verbose: print(f"early exit: found more than top_k ({self.top_k}) rule combinations") 
            #    break   #TODO: validate this. maybe iterate all and cut last??

            cur_rule_combinations += 1
        combis = copy.deepcopy(self.complexity)
        combis.pop(0)
        combis.pop(1)
        print(f"rule combinations: {combis}")
        
        return self.all_rules_with_relevance[:self.top_k], self.complexity
    
    
    def _extract_rule_combination_relevance(self, length: int):

        
        all_possible_rules_with_preliminary_scores = sorted(
            self.recombinations_w_preliminary_score(length), 
            key=lambda x: x[1], 
            reverse=True)

        rule_2_relevance = {}

        terminate = False
        best_relevance_so_far = ABSURDLY_LOW_VALUE  # initialize with an absurdly low value

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        while i < len(all_possible_rules_with_preliminary_scores) and not terminate:

            current_rule, current_preliminary_score = all_possible_rules_with_preliminary_scores[i]

            if self.verbose: print(f"\n\tComputing relevance for rule:" + str(current_rule))
            current_rule_relevance = self._compute_relevance_for_rule(flatten_list(current_rule), length=length)  # must flatten list of rules which are combined into one rule (one list of tuples)
            rule_2_relevance[tuple([tuple(rule) for rule in current_rule])] = current_rule_relevance
            if self.verbose: print("\tObtained relevance: " + str(current_rule_relevance))

            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = current_rule_relevance

            # early termination
            if current_rule_relevance > self.xsi:
                i += 1
                self.complexity["l.599>xsi"].append((self.sample_to_explain, current_rule_relevance)) 
                break

            # else, if the current relevance value is an improvement over the best relevance value seen so far, continue
            elif current_rule_relevance >= best_relevance_so_far:
                best_relevance_so_far = current_rule_relevance
                i += 1
                continue

            # else, if the window has not been filled yet, continue
            elif i < self.window_size:
                i += 1
                continue

            # else, use the average of the relevances in the window to assess the termination condition
            else:
                cur_avg_window_relevance = self._average(sliding_window)
                terminate_threshold = cur_avg_window_relevance / best_relevance_so_far
                random_value = random.random()
                terminate = random_value > terminate_threshold  # termination condition

                print("\n\tCurrent relevance " + str(current_rule_relevance))
                print("\tCurrent averaged window relevance " + str(cur_avg_window_relevance))
                print("\tMax relevance seen so far " + str(best_relevance_so_far))
                print("\tTerminate threshold:" + str(terminate_threshold))
                print("\tRandom value:" + str(random_value))
                print("\tTerminate:" + str(terminate))
                i += 1

        return rule_2_relevance
    

    def extract_rule_confidence(self, mode:str, rules:list[tuple[int, int, int]])->dict:
        if mode == "necessary":
            evidence = self.ruleset.necessary_row_ids
        elif mode == "sufficient":
            evidence = self.ruleset.sufficient_row_ids

        rule_2_preliminary_relevance = {}     #preliminary importance per rule

        for rule in rules:
            row_id = evidence[rule]
            row = self.ruleset.dataFrame.iloc[row_id]
            rule_2_preliminary_relevance[rule] = row[self.ruleset.order_col]

        self.sample_2_relevance = rule_2_preliminary_relevance
        return rule_2_preliminary_relevance
    

    def extract_rule_relevance_incrementally(self, rule_value_list):
        # start with most important rule
        cur_rule_length = 1
        current_set = []
        best_relevance_so_far = ABSURDLY_LOW_VALUE  # initialize with an absurdly low value
        total_length = len(rule_value_list)

        while cur_rule_length <= total_length and len(rule_value_list) > 0:
            new_rule, prelim_value = rule_value_list.pop(0)
            current_relevance = self._compute_relevance_for_rule \
                ([*current_set, *new_rule], cur_rule_length)     # unpack list and tuple with *
            if (current_relevance - best_relevance_so_far) > self.improvement_threshhold:
                current_set += new_rule
                best_relevance_so_far = current_relevance
                cur_rule_length += 1
                
                print(best_relevance_so_far, self._pretty_print_list(current_set))
            else:
                print(f"{current_relevance} --- skipped {self._pretty_print_list(new_rule)}")
                continue

        return [(tuple(current_set), best_relevance_so_far)], self.complexity


    def _compute_relevance_for_rule(self, nple_to_remove:list[tuple[int]], length:int) -> float:
        assert self.relevant_triples_set != None, "relevant_triples_set are None, please use setter before calling compute_relevance_for_rule()"

        if length in self.complexity:
            self.complexity[length] += 1
        else:
            self.complexity[length] = 1

        if type(nple_to_remove[0]) == int:
            nple_to_remove = [nple_to_remove]

        # convert the nple to remove into a list
        assert (len(nple_to_remove[0]) == 3), "nple_to_remove must be a list of triples"

        relevance, \
        original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
        base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
        pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, execution_time = \
            self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
                                          perspective=self.perspective,
                                          samples_to_remove=nple_to_remove,
                                          relevant_triples=self.relevant_triples_set)

        cur_line = ";".join(self.dataset.sample_to_fact(self.sample_to_explain)) + ";" + \
                   ";".join([";".join(self.dataset.sample_to_fact(x)) for x in nple_to_remove]) + ";" + \
                   str(original_target_entity_score) + ";" + \
                   str(original_target_entity_rank) + ";" + \
                   str(base_pt_target_entity_score) + ";" + \
                   str(base_pt_target_entity_rank) + ";" + \
                   str(pt_target_entity_score) + ";" + \
                   str(pt_target_entity_rank) + ";" + \
                   str(relevance) + ";" + \
                   str(execution_time)

        filename_1_details = self.output_prefix + "_10_output_details_" + str(length) + ".csv"

        #print(filename_1_details)
        with open(filename_1_details, "a") as output_file:
            output_file.writelines([cur_line + "\n"])

        return relevance
    
    
    def short_print_rule_relevance(self, rule_samples_with_relevance: dict|list, inplace=True):
        if type(rule_samples_with_relevance) == dict:
            rule_samples_with_relevance = sorted(rule_samples_with_relevance.items(), key=lambda x: x[1], reverse=True)
        result = ""
        for cur_rule_with_relevance in rule_samples_with_relevance:
            cur_rule_samples, cur_relevance = cur_rule_with_relevance
            if inplace:
                print(f"{round(float(cur_relevance), 3)} : {str(self.ruleset.extract_relations(cur_rule_samples, self.dataset))}")
            else:
                result += f"{round(float(cur_relevance), 3)} : {str(self.ruleset.extract_relations(cur_rule_samples, self.dataset))}; "
        return result


    def recombinations_w_preliminary_score(self, length):
        all_possible_rule_combinations = itertools.combinations(self.rules_to_remove, length)
        all_possible_rules_with_preliminary_scores = [(x, self._calc_preliminary_rule_score(tuple(flatten_list(x)))) 
                                                      for x in all_possible_rule_combinations]
        return all_possible_rules_with_preliminary_scores

    
       #TODO: probably not static, store preliminary rule score in self
    def _calc_preliminary_rule_score(self, rule: tuple[tuple[int, int, int]]) -> float:
        try:
            result = (numpy.sum([self.sample_2_relevance[x] for x in rule])/len(rule))   #avg importance
        except TypeError as e:
            if "list" in str(e):
                print(f"TypeError: list not hashable, must be tuple!")
            raise TypeError
        return result
    

    @staticmethod
    def _average(l: list):
        result = 0.0
        for item in l:
            result += float(item)
        return result / float(len(l))


    def _pretty_print_list(self, liste:list)->str:
        print_list = ""
        if type(liste[0])==int:
            liste = [liste]

        for item in liste:
            print_list += (f"{str(self.dataset.sample_to_fact(item))}")
        return str(print_list)
    
    def _pretty_print_dict(self, dictionary:dict)->str:
        print_dictionary = ""
        for key, value in dictionary.items():
            print_dictionary += (f"{value} : {str(self.dataset.sample_to_fact(key))}\n")
        return str(print_dictionary)
    
class EmptyResultException(Exception):
    pass

class RuleNotApplicableException(Exception):
    pass
