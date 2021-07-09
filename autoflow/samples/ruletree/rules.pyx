from datarule cimport DataWordRuleTree as CRuleTree
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cdef class RuleTree:
    cdef CRuleTree* ruleTree
    def __cinit__(self, rule_strs):
        self.ruleTree = new CRuleTree(r.encode('utf-8',errors='ignore') for r in rule_strs)
    def __dealloc__(self):
        del self.ruleTree
    def calc_scores(self,type_str,params):
        return self.ruleTree.calcScores(type_str,"\03".join(params).encode('utf-8',errors='ignore'))
    def get_score(self):
        return self.ruleTree.getScore()
    def reset(self):
        self.ruleTree.reset()
    def get_sub_scores(self):
        return self.ruleTree.getSubScores()
    def set_sub_scores(self, scores):
        self.ruleTree.setSubScores(scores)
    def print_tree(self):
        return self.ruleTree.printTree()
