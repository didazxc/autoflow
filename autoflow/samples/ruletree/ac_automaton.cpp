#include "ac_automaton.h"


AcAutomaton::AcAutomaton()
{
  root.fail = nullptr;
  root.isFinal = false;
}


AcAutomaton::~AcAutomaton()
{

  DeleteNode(&root);
}

//递归清理内存
void AcAutomaton::DeleteNode(Node *pNode)
{
  if (!pNode)return;

  for (auto iter = pNode->child.begin(); iter != pNode->child.end(); ++iter)
  {
    DeleteNode(iter->second);

    delete iter->second;
  }
}

//添加一个关键词
void AcAutomaton::AddString(const std::string &str)
{
  Node *p = &root;
  for (char cc:str)
  {
    auto iter = p->child.find(cc);
    Node *pnew;
    if (iter == p->child.end())
    {
      pnew = new node;
      pnew->achar = cc;
      pnew->fail = nullptr;
      pnew->isFinal = false;
      p->child.insert(std::unordered_map<char, node*>::value_type(cc, pnew));
    }
    else
    {
      pnew = iter->second;
    }
    p = pnew;
  }
  p->isFinal = true;
  p->word = str;
  p->utf8len = str.size();
}

//构造失败指针
void AcAutomaton::Build()
{
  std::queue<Node *> nodeQ;

  nodeQ.emplace(&root);

  while (!nodeQ.empty())
  {
    Node *cur = nodeQ.front();
    nodeQ.pop();


    for (auto iter = cur->child.begin(); iter != cur->child.end(); ++iter)
    {
      Node *subnode = iter->second;
      subnode->fail = &root;

      Node *q = cur->fail;

      nodeQ.emplace(subnode);

      while (q)
      {
        auto fpos = q->child.find(subnode->achar);
        if (fpos != q->child.end())
        {
          subnode->fail = fpos->second;
          break;
        }
        else
        {
          q = q->fail;
        }
      }
    }

  }

}

//查找str匹配的所有关键词
void AcAutomaton::Search(const std::string &str, std::unordered_set<std::string> &result)
{
  Node *p = &root;

  for (char cc:str)
  {
    auto fpos = p->child.find(cc);

    while (p != &root && (fpos == p->child.end()))
    {
      p = p->fail;
      fpos = p->child.find(cc);
    }

    if (fpos != p->child.end())
    {
      p = fpos->second;
    }
    else
    {
      continue; //从下一个字符开始匹配
    }


    Node *tmp = p;

    while (tmp)
    {
      if (tmp->isFinal)
      {
        result.emplace(tmp->word);
      }
      tmp = tmp->fail;
    }
  }
}


void AcAutomaton::SearchWithPos(const std::string &str, std::list<AcAutomatonHitPos> &result)
{
  Node *p = &root;
  int utf8pos = 0;
  for (char cc:str)
  {
    utf8pos++;
    auto fpos = p->child.find(cc);

    while (p != &root && (fpos == p->child.end()))
    {
      p = p->fail;
      fpos = p->child.find(cc);
    }

    if (fpos != p->child.end())
    {
      p = fpos->second;
    }
    else
    {
      continue; //从下一个字符开始匹配
    }


    Node *tmp = p;

    while (tmp)
    {
      if (tmp->isFinal)
      {
        result.emplace_back(std::make_tuple(tmp->word, utf8pos - tmp->utf8len));
      }
      tmp = tmp->fail;
    }
  }
}

bool AcAutomaton::HittedWords(const std::string &str)
{
  std::list<AcAutomatonHitPos> result;
  SearchWithPos(str, result);
  return !result.empty();
}


int get_utf_8_word_length(const char *str, int length)
{
  if (length < 1) {
    return 0;
  }
  char tmp = str[0];
  if (tmp > 0) {
    return 1;
  }

  int i = 0;
  do {
    tmp <<= 1;
    i++;
  } while (tmp < 0);

  return i > length ? length : i;
}
