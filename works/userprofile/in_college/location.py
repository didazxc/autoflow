import requests
import asyncio
import aiohttp
from autoflow.utils.geo.baidu_map_converter import Mercator_to_BD09, BD09_to_GCJ02
import logging
import os


def requests_test():
    url = "https://api.map.baidu.com"
    name = "中国政法大学"
    params = {
        "qt": "s",
        "c": 265,
        "wd": name,
        "rn": 10,
        "ie": "utf-8",
        "oue": 1,
        "fromproduct": "jsapi",
        "v": "2.1",
        "res": "api"
    }
    r = requests.get(url, params=params)
    write_out(parse(r.json()['content'], name))


def parse_single(college, name):
    bd_lng, bd_lat = Mercator_to_BD09(college['x'] / 100, college['y'] / 100)
    gc_lng, gc_lat = BD09_to_GCJ02(bd_lng, bd_lat)
    lng_lat = list(map(lambda x: str(x), [bd_lng, bd_lat, gc_lng, gc_lat]))
    alias = ','.join(map(lambda x: str(x), college.get('alias', [])))
    return ','.join([name, college['name']] + lng_lat + [alias, college.get('std_tag', '')])+'\n'


def parse(content: list, name: str) -> list:
    res = []
    college_name = name.split('\t', 1)[0].strip()
    index = college_name.find("（")
    if index:
        college_name = college_name[:index]
    try:
        for c in content:
            std_tag = c.get('std_tag', '')
            is_college = std_tag in ("教育培训;高等院校", "教育培训;成人教育")
            if not is_college:
                di_tag = c.get('di_tag', '')
                if std_tag in ("教育培训;其他", "教育培训;中学"):
                    is_college = '大学' in di_tag \
                                 or '高等院校' in di_tag \
                                 or c['name'].endswith('学院')
                if not is_college and college_name in c['name']:
                    is_college = '宿舍' in di_tag \
                                 or '食堂' in di_tag \
                                 or '图书馆' in di_tag \
                                 or '教学楼' in di_tag
            if is_college:
                res.append(parse_single(c, name))
        if len(res) == 0:
            for c in content:
                di_tag = c.get('di_tag', '')
                is_college = college_name in c['name'] and ('门' in di_tag or c.get('cp', '') == 'bus')
                if is_college:
                    res.append(parse_single(c, name))
    except Exception as e:
        logging.error(e)
    return res


def write_out(lines: list):
    if lines:
        if not os.path.exists('out.csv'):
            with open('out.csv', 'w', encoding='utf-8') as f:
                f.write("search_word,name,bd_lng,bd_lat,gc_lng,gc_lat,alias,std_tag\n")
                f.writelines(lines)
        else:
            with open('out.csv', 'a', encoding='utf-8') as f:
                f.writelines(lines)
        # with open('out.csv', 'r', encoding='utf-8') as f:
        #     s = {line.split(",")[0] for line in f}
        # logging.info(f"已插入条数：{len(s)}")


async def consumer(input_queue: asyncio.Queue):
    url = "https://api.map.baidu.com"
    timeout = aiohttp.ClientTimeout(total=300, connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            name, code = await input_queue.get()
            logging.debug('start: ' + name)
            college_name, city = name.split('\t')
            for w in ['宿舍 教学楼', '']:
                params = {
                    "qt": "s",
                    "c": code,
                    "wd": f"{college_name} {w} {city if code == 131 else ''}",
                    "rn": 50,  # 20 items number per page, 'pn: 0' is page number
                    "ie": "utf-8",
                    "oue": 1,
                    "fromproduct": "jsapi",
                    "v": "2.1",
                    "res": "api"
                }
                try:
                    async with session.get(url=url, params=params) as resp:
                        result = await resp.json(content_type='application/javascript;charset=utf-8')
                    logging.debug('get result: ' + name)
                except Exception as e:
                    await input_queue.put((name, code))
                    logging.debug(f'error: {name}\n{e}')
                else:
                    if 'content' in content:
                        content = result['content']
                        if content and 'code' in content[0]:
                            city = name.split('\t')[1].strip()
                            for c in content:
                                if c['name'].strip() == city:
                                    await input_queue.put((name, c['code']))
                                    break
                            logging.debug(f'change city: {name}-{city}')
                        else:
                            res = parse(content, name)
                            logging.debug(f'parse: {name}-{res}')
                            if len(res) > 0:
                                write_out(res)
                            # else:
                            #     write_out([name + '\n'])
                        break

            input_queue.task_done()


async def producer(input_queue: asyncio.Queue):
    try:
        with open("out.csv", "r", encoding='utf-8') as f:
            s = {line.split(",", 1)[0].strip() for line in f}
    except FileNotFoundError:
        s = {}
    res = []
    with open("colleges.csv", "r", encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name and name not in s:
                res.append(name)
    logging.info(f"已插入条数：{len(s)-1}, 剩余条数：{len(res)} {res}")
    for name in res:
        await input_queue.put((name, 131))
        # logging.debug('-- Thread1 -- insert: '+name)


async def async_main_fn():
    input_queue = asyncio.Queue()
    producer_task = asyncio.create_task(producer(input_queue))
    tasks = [asyncio.create_task(consumer(input_queue)) for _ in range(1)]
    await producer_task
    await input_queue.join()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    # requests_test()
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(async_main_fn())
