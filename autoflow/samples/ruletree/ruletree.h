#ifndef RULETREE_H
#define RULETREE_H

#include <string>
#include <vector>
#include <memory>
#include <stack>

using namespace std;

namespace ruletree {

    // Symbol

    class Symbol{
        public:
            const char symbol;
            const int priority;
            const bool unary;
        public:
            virtual int getScore(const vector<int>& scores);
            Symbol(const char& symbol_char):symbol(symbol_char),priority(0),unary(true){};
            Symbol(const char& symbol_char, const int& priority, const bool& unary):symbol(symbol_char),priority(priority),unary(unary){};
    };

    class OriginSymbol: public Symbol{
         public:
            OriginSymbol():Symbol('\01', 0, true){};
            int getScore(const vector<int>& scores);
    };

    class AndSymbol: public Symbol{
         public:
            AndSymbol(const char& symbol_char):Symbol(symbol_char, 1, false){};
            int getScore(const vector<int>& scores);
    };

    class OrSymbol: public Symbol{
        public:
            OrSymbol(const char& symbol_char):Symbol(symbol_char, 1, false){};
            int getScore(const vector<int>& scores);
    };

    class NotSymbol: public Symbol{
        public:
            NotSymbol(const char& symbol_char):Symbol(symbol_char, 2, true){};
            int getScore(const vector<int>& scores);
    };

    // Node

    class Node{
        public:
            virtual int getScore()=0;
            virtual void print(size_t depth=0)=0;
            virtual string getStr()=0;
    };

    class RuleNode: public Node{
        // 存放得分
        public:
            vector<int>* scores;
            const string rule_str;
            size_t start_index;
            size_t end_index;
        public:
            RuleNode(vector<int>* scores, const string& rule_str, const size_t& start_index, const size_t& end_index):scores(scores),rule_str(rule_str),start_index(start_index),end_index(end_index){};
            virtual int getScore();
            void print(size_t depth=0);
            string getStr();
    };

    class SymbolNode: public Node{
        // 存放逻辑连接符和其下的规则节点，并可获取得分
        public:
            const shared_ptr<Symbol> symbol;
            vector<shared_ptr<Node>> nodes;
            int min_value;
            int max_value;
        public:
            SymbolNode(const shared_ptr<Symbol>& symbol):symbol(symbol),min_value(INT_MIN),max_value(INT_MAX){};
            SymbolNode(const shared_ptr<Symbol>& symbol, const int& min_value, const int& max_value):symbol(symbol),min_value(min_value),max_value(max_value){};
            void addNode(shared_ptr<Node> node){this->nodes.emplace_back(node);};
            int getScore();
            // shared_ptr<Node> firstNode(){return this->nodes.front();};
            void print(size_t depth=0);
            string getStr();
            void setThreshold(const int& min_value,const int& max_value=INT_MAX){this->min_value=min_value;this->max_value=max_value;};
            vector<shared_ptr<RuleNode>> getRuleNodes();
    };

    // Rule

    class Rule{
        public:
            // 强规则字符创解析为规则节点，并在全局scores内存中申请一段作为该规则节点的scores内存
            virtual shared_ptr<RuleNode> parse(vector<int>& scores, const string& rule_str)=0;
            // 所有规则解析完之后的回调
            virtual void parse_end(){};
            // 计算得分到规则节点中
            virtual void calcScores(vector<int>& scores, const string& type_str, const vector<string>& params){};
            static void split(const string& s, const string& delim, vector<string>* ret);
            static string& trim(string &s);
    };

    class DigitRule: public Rule{
        public:
            shared_ptr<RuleNode> parse(vector<int>& scores, const string& rule_str);
    };

    // RuleTree

    class RuleTree{
        public:
            RuleTree(const vector<string>& rule_strs,
                        const vector<shared_ptr<Symbol>>& symbols,
                        const vector<shared_ptr<Rule>>& rules,
                        const char& BraceLeft,
                        const char& BraceRight, const char& AttributeBraceLeft, const char& AttributeBraceRight);
            RuleTree(const vector<string>& rule_strs);
            RuleTree(const vector<string>& rule_strs, const vector<shared_ptr<Rule>>& rules);
            vector<int> getScore();
            int setSubScores(vector<int> scores){this->scores.assign(scores.begin(),scores.end()); return 1;};
            vector<int> getSubScores(){return this->scores;};
            void reset(){fill(this->scores.begin(), this->scores.end(), 0);}
            int printTree(){for(auto node:this->rootNodes){node->print();} return 1;}
        public:
            vector<int> scores;
            vector<shared_ptr<RuleNode>> ruleNodes;
            vector<shared_ptr<SymbolNode>> rootNodes;
            const vector<shared_ptr<Rule>> rules;
            static const shared_ptr<Symbol> OriginSymbol;
        protected:
            const char BraceLeft;
            const char BraceRight;
            const char AttributeBraceLeftChar;
            const char AttributeBraceRightChar;
            const shared_ptr<SymbolNode> BraceLeftSymbolNode;
            const shared_ptr<SymbolNode> BraceRightSymbolNode;
            void init(const vector<string>& rule_strs, const vector<shared_ptr<Symbol>>& symbols, const vector<shared_ptr<Rule>>& rules);
        private:
            shared_ptr<SymbolNode> buildTree(string& rule_str,const vector<shared_ptr<Symbol>>& symbols,const vector<shared_ptr<Rule>>& rules);
            void merge_stack(stack<shared_ptr<SymbolNode>>& nodes_stack, const shared_ptr<SymbolNode>& next_node=NULL);
            void push_normal_symbol_and_rule_node(stack<shared_ptr<SymbolNode>>& nodes_stack, const shared_ptr<RuleNode>& p_rule, const shared_ptr<SymbolNode>& p_symbol_node);
            shared_ptr<RuleNode> parse_sub_rule_node(const vector<shared_ptr<Rule>>& rules, const string& rule_str, const int& start_index, const int& length);
    };

}

#endif