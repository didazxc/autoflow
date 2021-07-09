#include "ruletree.h"
#include <iostream>

using namespace std;

namespace ruletree {

    //Symbol

    int Symbol::getScore(const vector<int>& scores){
        /*cout<<"Base: ";
        for(int i : scores){cout<<i<<' ';}
        cout<<endl;*/
        return scores.front();
    }

    int OriginSymbol::getScore(const vector<int>& scores){
        /*cout<<"Origin: ";
        for(int i : scores){cout<<i<<' ';}
        cout<<endl;*/
        return scores.front();
    }

    int AndSymbol::getScore(const vector<int>& scores){
        /*cout<<"and: ";
        for(int i : scores){cout<<i<<' ';}
        cout<<endl;*/
        int min=scores[0];
        for(unsigned int i=1;i<scores.size();i++){
            if(min>scores[i]){min=scores[i];}
        }
        return min;
    }

    int OrSymbol::getScore(const vector<int>& scores){
        /*cout<<"or: ";
        for(int i : scores){cout<<i<<' ';}
        cout<<endl;*/
        int sum=0;
        for(int i : scores){sum+=i;}
        return sum;
    }

    int NotSymbol::getScore(const vector<int>& scores){
        /*cout<<"not: ";
        for(int i : scores){cout<<i<<' ';}
        cout<<endl;*/
        return scores.front()>0?0:1;
    }

    //SymbolNode

    int SymbolNode::getScore(){
        vector<int> res;
        for(shared_ptr<Node> node : nodes){
            int a=node->getScore();
            res.emplace_back(a);
        }
        int score = this->symbol->getScore(res);
        return score>=min_value && score<=max_value ? score : 0;
    }

    void SymbolNode::print(size_t depth){
        cout<<string(depth, '|')<<'('<<this->symbol->symbol<<")["<<min_value<<'-'<<max_value<<']'<<endl;
        for(shared_ptr<Node> node : nodes){
            node->print(depth+1);
        }
    }

    string SymbolNode::getStr(){
        string s;
        if(nodes.size()==1){
            s=symbol->symbol+nodes[0]->getStr();
        }else{
            for(size_t i=0;i<nodes.size();++i){
                shared_ptr<SymbolNode> pnode = dynamic_pointer_cast<SymbolNode>(nodes[i]);
                if(pnode==nullptr){
                    s += nodes[i]->getStr();
                }else{
                    s+="("+pnode->getStr()+")["+to_string(pnode->min_value)+"-"+to_string(pnode->max_value)+"]";
                }
                if(i<nodes.size()-1)s+=symbol->symbol;
            }
        }
        return s;
    }

    vector<shared_ptr<RuleNode>> SymbolNode::getRuleNodes(){
        vector<shared_ptr<RuleNode>> ruleNodes;
        for(shared_ptr<Node> node:nodes){
            shared_ptr<SymbolNode> snode = dynamic_pointer_cast<SymbolNode>(node);
            if(snode==nullptr){
                ruleNodes.emplace_back(static_pointer_cast<RuleNode>(node));
            }else{
                for(shared_ptr<RuleNode> p:snode->getRuleNodes()){
                    ruleNodes.emplace_back(p);
                }
            }
        }
        return ruleNodes;
    }

    //RuleNode

    void RuleNode::print(size_t depth){cout<<string(depth,'|')<<rule_str<<'('<<start_index<<','<<end_index<<')'<<endl;}

    int RuleNode::getScore(){return (*this->scores)[this->start_index];}

    string RuleNode::getStr(){return rule_str;}

    //Rule

    void Rule::split(const string& s, const string& delim, vector<string>* ret){
        size_t last = 0;
        size_t index=s.find_first_of(delim,last);
        while (index!=string::npos)
        {
            ret->emplace_back(s.substr(last,index-last));
            last=index+1;
            index=s.find_first_of(delim,last);
        }
        if (index-last>0)
        {
            ret->emplace_back(s.substr(last,index-last));
        }
    }

    string& Rule::trim(string &s){
        if (s.empty())
        {
            return s;
        }
        s.erase(0,s.find_first_not_of(" "));
        s.erase(s.find_last_not_of(" ") + 1);
        return s;
    }

    shared_ptr<RuleNode> DigitRule::parse(vector<int>& scores, const string& rule_str){
        size_t start_index;
        vector<string> ret;
        string delim = "=";
        Rule::split(rule_str, delim, &ret);
        if(Rule::trim(ret[0])=="digit"){
            start_index = scores.size();
            int score = atoi(ret[1].c_str());
            scores.emplace_back(score);
            return make_shared<RuleNode>(&scores, ret[1], start_index, scores.size());
        }
        return nullptr;
    }

    //RuleTree

    vector<int> RuleTree::getScore(){
        vector<int> res;
        for(shared_ptr<SymbolNode>& node:rootNodes){res.emplace_back(node->getScore());}
        return res;
    };

    void RuleTree::merge_stack(stack<shared_ptr<SymbolNode>>& nodes_stack, const shared_ptr<SymbolNode>& next_node_without_nodes){
        shared_ptr<SymbolNode> pre_node;
        shared_ptr<SymbolNode> value_node=nodes_stack.top();
        if(value_node==this->BraceLeftSymbolNode){
            nodes_stack.push(next_node_without_nodes);
            return;
        }
        nodes_stack.pop();
        while(! nodes_stack.empty()){
            pre_node = nodes_stack.top();
            if(pre_node==this->BraceLeftSymbolNode){
                if(next_node_without_nodes==this->BraceRightSymbolNode){
                    nodes_stack.pop();
                }
                break;
            }
            if(next_node_without_nodes==nullptr || pre_node->symbol->priority >= next_node_without_nodes->symbol->priority){
                pre_node->addNode(value_node);
                value_node=pre_node;
                nodes_stack.pop();
            }else{
                break;
            }
        }
        if(next_node_without_nodes!=nullptr && next_node_without_nodes!=this->BraceRightSymbolNode && (next_node_without_nodes->symbol!=value_node->symbol || next_node_without_nodes->min_value!=INT_MIN || next_node_without_nodes->max_value!=INT_MAX)){
            next_node_without_nodes->addNode(value_node);
            value_node = next_node_without_nodes;
        }

        nodes_stack.push(value_node);
    }

    void RuleTree::push_normal_symbol_and_rule_node(stack<shared_ptr<SymbolNode>>& nodes_stack, const shared_ptr<RuleNode>& p_rule, const shared_ptr<SymbolNode>& p_symbol_node){
        if(nodes_stack.empty()){
            p_symbol_node->addNode(p_rule);
            nodes_stack.push(p_symbol_node);
        }else{
            shared_ptr<SymbolNode> p_last = nodes_stack.top();
            if(p_last->symbol==p_symbol_node->symbol){
                p_last->addNode(p_rule);
                return;
            }
            if(p_last->symbol->priority<p_symbol_node->symbol->priority){
                p_symbol_node->addNode(p_rule);
                nodes_stack.push(p_symbol_node);
            }else{
                p_last->addNode(p_rule);
                merge_stack(nodes_stack, p_symbol_node);
            }
        }
    }

    shared_ptr<RuleNode> RuleTree::parse_sub_rule_node(const vector<shared_ptr<Rule>>& rules, const string& rule_str, const int& start_index, const int& length){
        shared_ptr<RuleNode> p_rule_node;
        string sub_rule_str = rule_str.substr(start_index, length);
        Rule::trim(sub_rule_str);
        if(sub_rule_str.length()==0) return nullptr;
        for(shared_ptr<Rule> p_rule : rules){
            p_rule_node = p_rule->parse(this->scores, sub_rule_str);
            if(p_rule_node!=nullptr){
                this->ruleNodes.emplace_back(p_rule_node);
                break;
            }
        }
        return p_rule_node;
    }

    shared_ptr<SymbolNode> RuleTree::buildTree(string& rule_str,const vector<shared_ptr<Symbol>>& symbols,const vector<shared_ptr<Rule>>& rules){
        stack<shared_ptr<SymbolNode>> nodes_stack;
        string rule_substr;
        shared_ptr<Rule> p_rule;
        shared_ptr<RuleNode> p_rule_node;
        shared_ptr<SymbolNode> p_symbol_node;
        int size=rule_str.length();
        int last_i=0;
        for(int i=0;i<size;++i){
            char c=rule_str[i];
            if(c==this->BraceLeft){
                if(i>0 && rule_str[i-1]=='\\'){
                    size-=1;
                    rule_str.erase(i-1,1);
                    continue;
                }
                last_i=i+1;
                nodes_stack.push(this->BraceLeftSymbolNode);
            }else if(c==this->BraceRight){
                if(i>0 && rule_str[i-1]=='\\'){
                    size-=1;
                    rule_str.erase(i-1,1);
                    continue;
                }
                if(last_i<i){
                    p_rule_node=this->parse_sub_rule_node(rules, rule_str, last_i, i-last_i);
                    p_symbol_node=nodes_stack.top();
                    if(p_symbol_node==this->BraceLeftSymbolNode){
                        p_symbol_node = make_shared<SymbolNode>(this->OriginSymbol);
                        nodes_stack.push(p_symbol_node);
                    }
                    p_symbol_node->addNode(p_rule_node);
                }
                last_i=i+1;
                merge_stack(nodes_stack, this->BraceRightSymbolNode);
            }else if(c==this->AttributeBraceLeftChar && i>0){
                if(i>0 && rule_str[i-1]=='\\'){
                    size-=1;
                    rule_str.erase(i-1,1);
                    continue;
                }

                size_t right_index = rule_str.find_first_of(this->AttributeBraceRightChar,i);
                while(rule_str[right_index-1]=='\\' and right_index!=string::npos){
                    size-=1;
                    rule_str.erase(right_index-1,1);
                    right_index = rule_str.find_first_of(this->AttributeBraceRightChar,right_index);
                }

                if(rule_str[i-1]==this->BraceRight){
                    vector<string> ret;
                    Rule::split(rule_str.substr(i+1,right_index),"-", &ret);
                    nodes_stack.top()->setThreshold(atoi(ret[0].c_str()),ret.size()>=2?atoi(ret[1].c_str()):INT_MAX);
                    last_i = right_index+1;
                }
                i = right_index;
            }else{
                for(shared_ptr<Symbol> s : symbols){
                    if(s->symbol==c){
                        if(i>0 && rule_str[i-1]=='\\'){
                            size-=1;
                            rule_str.erase(i-1,1);
                            break;
                        }
                        p_symbol_node = make_shared<SymbolNode>(s);
                        if(s->unary){
                            nodes_stack.push(p_symbol_node);
                        }else{
                            p_rule_node=last_i<i?this->parse_sub_rule_node(rules, rule_str, last_i, i-last_i):nullptr;
                            if(p_rule_node!=nullptr){
                                push_normal_symbol_and_rule_node(nodes_stack, p_rule_node, p_symbol_node);
                            }else{
                                merge_stack(nodes_stack, p_symbol_node);
                            }
                        }
                        last_i=i+1;
                        break;
                    }
                }
            }
        }
        if(last_i<size){
            p_rule_node=this->parse_sub_rule_node(rules, rule_str, last_i, size-last_i);
            if(nodes_stack.empty()){
                p_symbol_node = make_shared<SymbolNode>(this->OriginSymbol);
                nodes_stack.push(p_symbol_node);
            }else{
                p_symbol_node = nodes_stack.top();
            }
            p_symbol_node->addNode(p_rule_node);
        }
        merge_stack(nodes_stack);
        return nodes_stack.top();
    }

    void RuleTree::init(const vector<string>& rule_strs,const vector<shared_ptr<Symbol>>& symbols,const vector<shared_ptr<Rule>>& rules){
        for(string rule_str:rule_strs){
            rootNodes.emplace_back(this->buildTree(rule_str, symbols, rules));
        }
        for(shared_ptr<Rule> p_rule : rules){
            p_rule->parse_end();
        }
    }

    const shared_ptr<Symbol> RuleTree::OriginSymbol=make_shared<Symbol>('\01');

    RuleTree::RuleTree(const vector<string>& rule_strs,
                        const vector<shared_ptr<Symbol>>& symbols,
                        const vector<shared_ptr<Rule>>& rules,
                        const char& braceLeft,
                        const char& braceRight,
                        const char& attributeBraceLeft,
                        const char& attributeBraceRight):
                                        rules(rules),
                                        BraceLeft(braceLeft),
                                        BraceRight(braceRight),
                                        AttributeBraceLeftChar(attributeBraceLeft),
                                        AttributeBraceRightChar(attributeBraceRight),
                                        BraceLeftSymbolNode(make_shared<SymbolNode>(make_shared<Symbol>(braceLeft))),
                                        BraceRightSymbolNode(make_shared<SymbolNode>(make_shared<Symbol>(braceRight)))
    {
        this->init(rule_strs, symbols, rules);
    }

    RuleTree::RuleTree(const vector<string>& rule_strs):
                                        rules(vector<shared_ptr<Rule>>{make_shared<DigitRule>()}),
                                        BraceLeft('('),
                                        BraceRight(')'),
                                        AttributeBraceLeftChar('\15'),
                                        AttributeBraceRightChar('\14'),
                                        BraceLeftSymbolNode(make_shared<SymbolNode>(make_shared<Symbol>('('))),
                                        BraceRightSymbolNode(make_shared<SymbolNode>(make_shared<Symbol>(')')))
    {
        vector<shared_ptr<Symbol>> symbols{make_shared<AndSymbol>('&'),make_shared<OrSymbol>('|'),make_shared<NotSymbol>('!')};
        this->init(rule_strs, symbols, this->rules);
    }

    RuleTree::RuleTree(const vector<string>& rule_strs, const vector<shared_ptr<Rule>>& rules):
                                        rules(rules),
                                        BraceLeft('('),
                                        BraceRight(')'),
                                        AttributeBraceLeftChar('\15'),
                                        AttributeBraceRightChar('\14'),
                                        BraceLeftSymbolNode(make_shared<SymbolNode>(make_shared<Symbol>('('))),
                                        BraceRightSymbolNode(make_shared<SymbolNode>(make_shared<Symbol>(')')))
    {
        vector<shared_ptr<Symbol>> symbols{make_shared<AndSymbol>('&'),make_shared<OrSymbol>('|'),make_shared<NotSymbol>('!')};
        this->init(rule_strs, symbols, rules);
    }



}