from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "ac_automaton.h":
    pass

cdef extern from "ac_automaton.cpp":
    pass

cdef extern from "ruletree.h":
    pass

cdef extern from "ruletree.cpp":
    pass

cdef extern from "datarule.cpp":
    pass

# Declare the class with cdef
cdef extern from "datarule.h" namespace "ruletree":
    cdef cppclass DataWordRuleTree:
        DataWordRuleTree(vector[string]) except +
        vector[int] getScore() except +
        vector[int] getSubScores() except +
        int setSubScores(vector[int]) except +
        int calcScores(string, string) except +
        int printTree() except +
        void reset() except +
