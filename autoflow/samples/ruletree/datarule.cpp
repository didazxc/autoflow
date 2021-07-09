#include "datarule.h"

namespace ruletree {

    shared_ptr<RuleNode> WordRule::parse(vector<int>& scores, const string& rule_str){
        return make_shared<RuleNode>(&scores, rule_str, 0, 0);
    }

    void DataWordRule::insertKwMap(shared_ptr<RuleNode>& pRuleNode, vector<int>& scores,const vector<string>& apps, const string& query){
        pRuleNode->scores = &scores;
        pRuleNode->start_index = scores.size();
        pRuleNode->end_index = pRuleNode->start_index+1;
        string word = pRuleNode->getStr();
        this->ac.AddString(word);
        for(string app:apps){
            string kw = app+":"+query+":"+word;
            this->kw_map[kw].emplace_back(ScoreIndex(&scores, pRuleNode->start_index));
        }
        scores.emplace_back(0);
    }

    void DataWordRule::insertSMap(shared_ptr<Node>& node, vector<int>& scores,const vector<string>& apps, const string& query){
        shared_ptr<SymbolNode> pnode = dynamic_pointer_cast<SymbolNode>(node);
        size_t scores_index = scores.size();
        scores.emplace_back(0);
        if(pnode->min_value==INT_MIN && pnode->max_value==INT_MAX){
            node=make_shared<RuleNode>(&scores,pnode->getStr(),scores_index, scores_index+1);
        }else{
            shared_ptr<SymbolNode> tmp_node=make_shared<SymbolNode>(RuleTree::OriginSymbol, pnode->min_value, pnode->max_value);
            tmp_node->addNode(make_shared<RuleNode>(&scores,pnode->getStr(),scores_index, scores_index+1));
            node=tmp_node;
            pnode->min_value=INT_MIN;
            pnode->max_value=INT_MAX;
        }
        // insert pnode to s_map, maybe can use pnode->getStr() as key for distinct
        this->s_map.emplace(pnode,ScoreIndex(&scores, scores_index));
        // insert ruleNode of pnode to kw_map
        for(shared_ptr<RuleNode> rnode:pnode->getRuleNodes()){
            this->insertKwMap(rnode, this->s_scores, apps, query);
        }
    }

    shared_ptr<RuleTree> DataWordRule::parseSubRuleTree(vector<int>& scores, const string& rule_str){
        // apps
        vector<string> apps;
        string delim = ":";
        size_t last = 0;
        size_t index=rule_str.find_first_of(delim,last);
        Rule::split(rule_str.substr(last,index-last), ",", &apps);
        for(size_t i=0;i<apps.size();++i){Rule::trim(apps[i]);}
        if(apps.empty())apps.emplace_back("all");
        // query
        string query;
        last=index+1;
        index=rule_str.find_first_of(delim,last);
        query=rule_str.substr(last,index-last);
        Rule::trim(query);
        if(query=="0"){
            this->onlyQueryMode = false;
        }
        // sub_rule
        const char orSymbol=',';
        vector<shared_ptr<Symbol>> symbols{make_shared<PlusSymbol>('+'),make_shared<OrSymbol>(orSymbol),make_shared<MinusSymbol>('-')};
        shared_ptr<RuleTree> pRuleTree = make_shared<RuleTree>(vector<string>{rule_str.substr(index+1)},symbols,vector<shared_ptr<Rule>>{make_shared<WordRule>()},'{','}','[',']');
        shared_ptr<SymbolNode> pSymbolNode = pRuleTree->rootNodes.front();
        if(pSymbolNode->symbol->symbol==orSymbol){
            for(shared_ptr<Node>& node:pSymbolNode->nodes){
                shared_ptr<SymbolNode> pnode = dynamic_pointer_cast<SymbolNode>(node);
                if(pnode==NULL || pnode->symbol==pRuleTree->OriginSymbol){
                    shared_ptr<RuleNode> pRuleNode;
                    if(pnode==NULL)pRuleNode = static_pointer_cast<RuleNode>(node);
                    else pRuleNode = static_pointer_cast<RuleNode>(pnode->nodes.front());
                    this->insertKwMap(pRuleNode, scores, apps, query);
                }else{
                    this->insertSMap(node, scores, apps, query);
                }
            }
        }else{
            if(pSymbolNode->symbol==pRuleTree->OriginSymbol){
                shared_ptr<RuleNode> pRuleNode = static_pointer_cast<RuleNode>(pSymbolNode->nodes.front());
                this->insertKwMap(pRuleNode, scores, apps, query);
            }else{
                shared_ptr<Node> node = pSymbolNode;
                this->insertSMap(node, scores, apps, query);
            }
        }
        return pRuleTree;
    }

    shared_ptr<RuleNode> DataWordRule::parse(vector<int>& scores, const string& rule_str){
        int start_index;
        vector<string> ret;
        string delim = "=";
        Rule::split(rule_str, delim, &ret);
        if(Rule::trim(ret[0])!="data_word") return NULL;
        start_index = scores.size();
        shared_ptr<RuleTree> pRuleTree=this->parseSubRuleTree(scores, ret[1]);
        return make_shared<DataWordRuleNode>(pRuleTree, &scores, rule_str, start_index, scores.size());
    }

    void DataWordRule::parse_end(){
        this->ac.Build();
    };

    void DataWordRule::calcScores(vector<int>& scores,const string& type_str, const vector<string>& params){
        if(type_str!="data") return;
        if(this->onlyQueryMode){
            this->calcScoresOnlyQuery(scores,type_str,params);
            return;
        }
        // 1.ruletrees reset 0
        this->resetScores();
        // 2.ac(params) update scores by kw_map
        string app;
        bool query;
        string sub_str;
        size_t index;
        size_t last;
        for(string line:params){
            index = line.find_first_of(':');
            app=line.substr(0, index);
            last = line.find_first_of(':', index+1)+1;
            index = line.size()-2;
            if(line[index]==':' && line[index+1]=='1'){
                query = true;
                sub_str=line.substr(last, index-last);
            }else{
                query = false;
                sub_str=line.substr(last);
            }
            list<AcAutomatonHitPos> result;
            this->ac.SearchWithPos(sub_str, result);
            unordered_map<string,vector<ScoreIndex>>::const_iterator got;
            for(AcAutomatonHitPos& hitPos : result){
                string word;
                int pos;
                std::tie(word, pos) = hitPos;
                vector<string> kws;
                if(query){
                    kws = vector<string>{app+":0:"+word,app+":1:"+word,"all:0:"+word,"all:1:"+word};
                }else{
                    kws = vector<string>{app+":0:"+word,"all:0:"+word};
                }
                for(string kw:kws){
                    got = this->kw_map.find(kw);
                    if(got!=kw_map.end()){
                        for(ScoreIndex x:got->second){
                            x.addScore(1);
                        }
                    }
                }
            }
        }
        // 3.ruletrees.calcScores and update global scores
        for ( auto it = this->s_map.begin(); it != this->s_map.end(); ++it ){
            it->second.addScore(it->first->getScore());
        }
    }

    void DataWordRule::calcScoresOnlyQuery(vector<int>& scores,const string& type_str, const vector<string>& params){
        // 1.ruletrees reset 0
        this->resetScores();
        // 2.ac(params) update scores by kw_map
        string app;
        string sub_str;
        size_t index;
        size_t last;
        for(string line:params){
            index = line.find_first_of(':');
            app=line.substr(0, index);
            last = line.find_first_of(':', index+1)+1;
            index = line.size()-2;
            if(line[index]==':' && line[index+1]=='1'){
                sub_str=line.substr(last, index-last);
            }else{
                continue;
            }
            list<AcAutomatonHitPos> result;
            this->ac.SearchWithPos(sub_str, result);
            unordered_map<string,vector<ScoreIndex>>::const_iterator got;
            for(AcAutomatonHitPos& hitPos : result){
                string word;
                int pos;
                std::tie(word, pos) = hitPos;
                for(string kw:vector<string>{app+":1:"+word,"all:1:"+word}){
                    got = this->kw_map.find(kw);
                    if(got!=kw_map.end()){
                        for(ScoreIndex x:got->second){
                            x.addScore(1);
                        }
                    }
                }
            }
        }
        // 3.ruletrees.calcScores and update global scores
        for ( auto it = this->s_map.begin(); it != this->s_map.end(); ++it ){
            it->second.addScore(it->first->getScore());
        }
    }

    shared_ptr<RuleNode> AppsDaysRule::parse(vector<int>& scores, const string& rule_str){
        return this->parseByKey(scores, rule_str, "apps_days");
    }

    shared_ptr<RuleNode> AppsDaysRule::parseByKey(vector<int>& scores, const string& rule_str, const string& key){
        int start_index;
        vector<string> ret;
        string delim = "=";
        Rule::split(rule_str, delim, &ret);
        if(Rule::trim(ret[0])!=key) return NULL;
        start_index = scores.size();
        shared_ptr<RuleTree> pRuleTree=this->parseSubRuleTree(scores, ret[1]);
        return make_shared<DataWordRuleNode>(pRuleTree, &scores, rule_str, start_index, scores.size());
    }

    void AppsDaysRule::insertKwMap(shared_ptr<RuleNode>& pRuleNode, vector<int>& scores){
        pRuleNode->scores = &scores;
        pRuleNode->start_index = scores.size();
        pRuleNode->end_index = pRuleNode->start_index+1;
        string kw = pRuleNode->getStr();
        unordered_map<string,vector<ScoreIndex>>::iterator got = this->kw_map.find(kw);
        if(got==this->kw_map.end()){
            this->kw_map.emplace(kw,vector<ScoreIndex>{ScoreIndex(&scores, pRuleNode->start_index)});
        }else{
            got->second.emplace_back(ScoreIndex(&scores, pRuleNode->start_index));
        }
        scores.emplace_back(0);
    }

    shared_ptr<RuleTree> AppsDaysRule::parseSubRuleTree(vector<int>& scores, const string& rule_str){
        vector<shared_ptr<Symbol>> symbols{make_shared<PlusSymbol>('+'),make_shared<OrSymbol>(','),make_shared<MinusSymbol>('-')};
        shared_ptr<RuleTree> pRuleTree = make_shared<RuleTree>(vector<string>{rule_str},symbols,vector<shared_ptr<Rule>>{make_shared<WordRule>()},'{','}','[',']');
        for(shared_ptr<RuleNode>& pnode:pRuleTree->ruleNodes){
            this->insertKwMap(pnode, scores);
        }
        return pRuleTree;
    }

    void AppsDaysRule::calcScores(vector<int>& scores, const string& type_str, const vector<string>& params){
        if(this->kw_map.empty())return;
        if(type_str=="apps"){
            size_t index;
            size_t last;
            string app;
            string freq_str;
            for(string app_str:params){
                // applist -- app:freq:dur:is_sys
                index = app_str.find_first_of(':');
                app = app_str.substr(0, index);
                last = index+1;
                index = app_str.find_first_of(':', last);
                freq_str = app_str.substr(index, last);
                if(freq_str.length()>0 && atoi(freq_str.c_str())>0){
                    unordered_map<string,vector<ScoreIndex>>::const_iterator got = this->kw_map.find(app);
                    if(got!=kw_map.end()){
                        for(ScoreIndex x:got->second){
                            x.addScore(1);
                        }
                    }
                }
            }
        }else if(type_str=="data" || type_str=="data_query"){
            unordered_set<string> inserted_apps;
            string app;
            for(string app_str:params){
                app = app_str.substr(0, app_str.find_first_of(':'));
                // data -- app:timestamp:corpus:is_query
                if(inserted_apps.find(app)==inserted_apps.end()){
                    unordered_map<string,vector<ScoreIndex>>::const_iterator got = this->kw_map.find(app);
                    if(got!=kw_map.end()){
                        inserted_apps.emplace(app);
                        for(ScoreIndex x:got->second){
                            x.addScore(1);
                        }
                    }
                }
            }
        }
    }

    shared_ptr<RuleNode> AppsFreqRule::parse(vector<int>& scores, const string& rule_str){
        return this->parseByKey(scores, rule_str, "apps_freq");
    }

    void AppsFreqRule::calcScores(vector<int>& scores, const string& type_str, const vector<string>& params){
        if(this->kw_map.empty())return;
        if(type_str=="apps"){
            size_t index;
            size_t last;
            string app;
            string freq_str;
            for(string app_str:params){
                index = app_str.find_first_of(':');
                app = app_str.substr(0, index);
                last = index+1;
                index = app_str.find_first_of(':', last);
                freq_str = app_str.substr(index, last);
                // applist -- app:freq:dur:is_sys
                if(freq_str.length()>0){
                    int freq=atoi(freq_str.c_str());
                    unordered_map<string,vector<ScoreIndex>>::const_iterator got = this->kw_map.find(app);
                    if(got!=kw_map.end()){
                        for(ScoreIndex x:got->second){
                            x.addScore(freq);
                        }
                    }
                }
            }
        }else if(type_str=="data" || type_str=="data_query"){
            string app;
            for(string app_str:params){
                app = app_str.substr(0, app_str.find_first_of(':'));
                // data -- app:timestamp:corpus:is_query
                unordered_map<string,vector<ScoreIndex>>::const_iterator got = this->kw_map.find(app);
                if(got!=kw_map.end()){
                    for(ScoreIndex x:got->second){
                        x.addScore(1);
                    }
                }
            }
        }
    }

    shared_ptr<RuleNode> AppsListRule::parse(vector<int>& scores, const string& rule_str){
        return this->parseByKey(scores, rule_str, "apps_list");
    }

    void AppsListRule::calcScores(vector<int>& scores, const string& type_str, const vector<string>& params){
        if(this->kw_map.empty())return;
        if(type_str=="apps" || type_str=="data" || type_str=="data_query"){
            for(string app_str:params){
                // applist -- app:freq:dur:is_sys; data -- app:timestamp:corpus:is_query
                string app = app_str.substr(0, app_str.find_first_of(':'));
                unordered_map<string,vector<ScoreIndex>>::const_iterator got = this->kw_map.find(app);
                if(got!=kw_map.end()){
                    for(ScoreIndex x:got->second){
                        x.setScore(1);
                    }
                }
            }
        }
    }

    DataWordRuleTree::DataWordRuleTree(const vector<string>& rule_strs):
        RuleTree(rule_strs,vector<shared_ptr<Rule>>{
            make_shared<DataWordRule>(),
            make_shared<AppsDaysRule>(),
            make_shared<AppsFreqRule>(),
            make_shared<AppsListRule>()}){}

    int DataWordRuleTree::calcScores(const string& type_str, const string& params){
        vector<string> res;
        Rule::split(params,"\03", &res);
        for(shared_ptr<Rule> rule:this->rules){
            rule->calcScores(this->scores, type_str, res);
        }
        return 0;
    }

}

