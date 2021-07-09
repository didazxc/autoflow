from ruletree.rules import RuleTree
import time


def test0():
    r = RuleTree(b'digit=13|((digit=10)&digit=22)|!digit=-1')
    r = RuleTree(b'digit=13|(digit=10&digit=22)')
    r = RuleTree(b'data_word=app1,app2:0:{k1}[1-3],{k2+k3}[0-2]|apps_days={app1}[6],app2')
    r = RuleTree(b'data_word=app1,app2:0:k1,k1')
    r = RuleTree(b'(data_word=app1,app2:1:k1[1-3],{k2+k3}[1-2] | ( data_word = app1,app3:0:k2,k1+k3 )\15 5 \14)')
    r.calc_scores(b"data", [b"app1:0000:k1:1", b"app1:0000:k1,k2,k3"])
    r.calc_scores(b"apps", [b"app1:5:1:1", b"app1:0:0:0"]*10000)


def test1(s:RuleTree):
    datas = ['app1:123:ak0k0k0k0k11k11k12字符:1', 'app4:123:k0k0k0k0k11k10字符', 'app4:123:k0字符']
    start_1 = time.perf_counter_ns()
    s.reset()
    s.calc_scores(b'data', datas)
    # s.calc_scores(b'apps', datas)
    print(s.get_sub_scores())
    print(s.get_score())
    s.reset()
    print(s.get_sub_scores())
    # s.print_tree()
    print(s.get_score())
    #print(datas, '->', list(zip(my_tags, s.get_score())), time.perf_counter_ns() - start_1)


def test2(s:RuleTree):
    datas = ['app4:123:k0k0k0k0k11k11' + '两个字 符' * 10000]
    start_2 = time.perf_counter()
    s.reset()
    s.calc_scores(b'data', datas)
    sub_scores = s.get_sub_scores()
    print(sub_scores)
    # s.reset()
    print('len(str)=10000 ->', list(zip(my_tags, s.get_score())), time.perf_counter() - start_2)

    datas = [f'app{i}'+':123:k0k0k0k0k11k11' + '两个字 符' for i in range(10000)]
    start_3 = time.perf_counter()
    s.reset()
    s.calc_scores(b'data', datas)
    print('len(datas)=', len(datas), ' ->', list(zip(my_tags, s.get_score())), time.perf_counter() - start_3)

    datas = [f'app{i}'+':123:k0k0k0k0k11k11' + '两个字 符:1' for i in range(10000)]
    start_4 = time.perf_counter()
    s.reset()
    s.calc_scores(b'data', datas)
    print('len(datas)=', len(datas), ' ->', list(zip(my_tags, s.get_score())), time.perf_counter() - start_4)
    a = s.get_sub_scores()
    print(a)
    s.reset()
    s.set_sub_scores(a)
    print(list(zip(my_tags, s.get_score())))


def test3():
    import json
    with open('/search/odin/zxc/autoflow/works/ams/rules.json', 'r') as f:
        rules: dict = json.load(f)
    tag_rules = list(rules.values())
    print(tag_rules)
    s = RuleTree(tag_rules)
    print(tag_rules)
    s.print_tree()


if __name__ == '__main__':
    my_tags = ['tag1', 'tag2', 'tag3', 'tag4']
    my_tag_rules = ['data_word=app1,app2:1:字 符,kwd2+k3+k4-k5,{k6,k7,k8}[5-6],{k9}[3-4],{k10}[3],{k11}[-4] '
                    '& ( data_word=app1,app2:1:字 符,kwd2+k3+k4-k5,{k6,k7,k8}[5-6],{k9}[3-4],{k10}[3],{k11}[-4] '
                    '| data_word=app3,app4:1:字 符,kwd2+k3+k4-k5,{k6,k7,k8}[5-6],{k9}[3-4],{k10}[3],{k11}[-4] ) '
                    '& !data_word=app2,app4:1:字 符,kwd2+k3+k4-k5,{k6,k7,k8}[5-6],{k9}[3-4],{k10}[3],{k11}[-4]',
                    'apps_freq=app4', 'data_word=all:1:字 符', 'data_word='+','.join(f'app{i}' for i in range(100000))+':1:字 符']
    start = time.perf_counter_ns()
    rule_tree = RuleTree(my_tag_rules)
    print('build time: ', time.perf_counter_ns() - start)
    # rule_tree.print_tree()

    test2(rule_tree)
