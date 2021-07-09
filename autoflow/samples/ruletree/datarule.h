#ifndef RULETREE_DATAWORDRULE_H
#define RULETREE_DATAWORDRULE_H

#include "ac_automaton.h"
#include "ruletree.h"

using namespace std;

namespace ruletree {

    struct ScoreIndex{
        vector<int>* scores;
        size_t index;
        ScoreIndex(vector<int>* scores,size_t index):scores(scores),index(index){};
        void addScore(const int& x){(*(scores))[index] += x;}
        void setScore(const int& x){(*(scores))[index] = x;}
    };

    class MinusSymbol: public Symbol{
        public:
            MinusSymbol(const char& symbol_char):Symbol(symbol_char, 2, false){};
            int getScore(const vector<int>& scores){
                int scores_size=scores.size();
                for(int i=1;i<scores_size;++i){
                    if(scores[i]>0)return 0;
                }
                return scores[0];
            };
    };

    class PlusSymbol: public Symbol{
         public:
            PlusSymbol(const char& symbol_char):Symbol(symbol_char, 2, false){};
            int getScore(const vector<int>& scores){
                int min=scores[0];
                for(unsigned int i=1;i<scores.size();i++){
                    if(min>scores[i]){min=scores[i];}
                }
                return min;
            };
    };

    class WordRule: public Rule{
        public:
            shared_ptr<RuleNode> parse(vector<int>& scores, const string& rule_str);
    };

    class DataWordRule: public Rule{
        public:
            shared_ptr<RuleNode> parse(vector<int>& scores, const string& rule_str);
            void parse_end();
            void calcScores(vector<int>& scores, const string& type_str, const vector<string>& params);
        protected:
            // 将词汇规则拆分为最小单元子规则，存储于此，方便汇总
            unordered_map<string,vector<ScoreIndex>> kw_map;
            AcAutomaton ac;
        private:
            bool onlyQueryMode{true};
            // 将词汇规则按逗号拆分出最外层的子规则，如果是RuleNode直接存储于kw_map，如果是SymbolNode则继续拆分存储，同时将该SymbolNode存储于此
            unordered_map<shared_ptr<SymbolNode>,ScoreIndex> s_map;
            // SymbolNode内部统一将得分存储于s_scores内存,SymbolNode的总得分存于scores中
            vector<int> s_scores;
            void resetScores(){fill(this->s_scores.begin(), this->s_scores.end(), 0);}
            void insertKwMap(shared_ptr<RuleNode>& pRuleNode, vector<int>& scores,const vector<string>& apps, const string& query);
            void insertSMap(shared_ptr<Node>& node, vector<int>& scores,const vector<string>& apps, const string& query);
            shared_ptr<RuleTree> parseSubRuleTree(vector<int>& scores, const string& rule_str);
            void calcScoresOnlyQuery(vector<int>& scores, const string& type_str, const vector<string>& params);
    };

    class DataWordRuleNode: public RuleNode{
        public:
            DataWordRuleNode(const shared_ptr<RuleTree>& pRuleTree, vector<int>* scores, const string& rule_str, const size_t& start_index, const size_t& end_index):RuleNode(scores, rule_str, start_index, end_index),pRuleTree(pRuleTree){};
            int getScore(){return pRuleTree->getScore().front();};
        protected:
            const shared_ptr<RuleTree> pRuleTree;
    };

    class AppsDaysRule: public Rule{
        public:
            shared_ptr<RuleNode> parse(vector<int>& scores, const string& rule_str);
            void calcScores(vector<int>& scores, const string& type_str, const vector<string>& params);
        protected:
            shared_ptr<RuleNode> parseByKey(vector<int>& scores, const string& rule_str, const string& key);
            unordered_map<string,vector<ScoreIndex>> kw_map;
            shared_ptr<RuleTree> parseSubRuleTree(vector<int>& scores, const string& rule_str);
        private:
            void insertKwMap(shared_ptr<RuleNode>& pRuleNode, vector<int>& scores);
    };

    class AppsFreqRule: public AppsDaysRule{
        public:
            shared_ptr<RuleNode> parse(vector<int>& scores, const string& rule_str);
            void calcScores(vector<int>& scores, const string& type_str, const vector<string>& params);
    };

    class AppsListRule: public AppsDaysRule{
        public:
            shared_ptr<RuleNode> parse(vector<int>& scores, const string& rule_str);
            void calcScores(vector<int>& scores, const string& type_str, const vector<string>& params);
    };

    class DataWordRuleTree: public RuleTree{
        public:
            DataWordRuleTree(const vector<string>& rule_str);
            int calcScores(const string& type_str, const string& params);
    };


}

#endif