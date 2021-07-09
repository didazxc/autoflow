#pragma once

#include <string>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <list>

using AcAutomatonHitPos = std::tuple<std::string, int>;

class AcAutomaton
{

public:

  void AddString(const std::string &str);//添加关键词

  void Build(); //所有关键词添加完成后需要执行一次Build

  void Search(const std::string &str, std::unordered_set<std::string> &result);//查找str中命中的所有关键词

  void SearchWithPos(const std::string &str, std::list<AcAutomatonHitPos> &result);

  bool HittedWords(const std::string &str);

  AcAutomaton();
  ~AcAutomaton();

private:

  struct node;

  typedef struct node {

    char achar; //当前的字符

    struct node *fail; //失败指针

    std::unordered_map<char, node*> child;//next结点的map

    std::string word; //如果是终结点，word对应就是一个关键词，否则是空

    bool isFinal;// 是否是终结点

    int utf8len;// 如果是终结点， 对应word的长度
  }Node;


  void DeleteNode(Node *pNode);

  Node root;

};

int get_utf_8_word_length(const char *str, int length);
