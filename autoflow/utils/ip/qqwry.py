#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: autoflow
@Date: 6/25/21 9:48 PM
@Desc:
from https://github.com/animalize/qqwry-python3
"""
import os
import array
import bisect
import struct
import socket
import logging
from typing import Tuple, Union

__all__ = ('QQwry',)

logger = logging.getLogger(__name__)


def int3(data, offset):
    return data[offset] + (data[offset + 1] << 8) + \
           (data[offset + 2] << 16)


def int4(data, offset):
    return data[offset] + (data[offset + 1] << 8) + \
           (data[offset + 2] << 16) + (data[offset + 3] << 24)


class QQwry:
    def __init__(self, filename: str = None, loadindex: bool = True) -> None:
        self.provinces = ['北京', '上海', '天津', '重庆', '香港', '澳门',
                          '广西', '宁夏', '新疆', '西藏',
                          '内蒙古', '黑龙江',
                          '辽宁', '吉林', '河北', '山西',
                          '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '海南',
                          '四川', '贵州', '云南', '陕西', '甘肃', '青海', '台湾']
        self.clear()
        if filename is None:
            this_dir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(this_dir, 'qqwry.dat')
        self.load_file(filename, loadindex)

    def clear(self) -> None:
        '''清空加载的数据，再次调用.load_file()时不必执行.clear()。'''
        self.idx1 = None
        self.idx2 = None
        self.idxo = None

        self.data = None
        self.index_begin = -1
        self.index_end = -1
        self.index_count = -1

        self.__fun = None

    def load_file(self, filename: Union[str, bytes], loadindex: bool = True) -> bool:
        '''加载qqwry.dat文件。成功返回True，失败返回False。
        参数filename可以是qqwry.dat的文件名（str类型），也可以是bytes类型的文件内容。'''
        self.clear()

        if type(filename) == bytes:
            self.data = buffer = filename
            filename = 'memory data'
        elif type(filename) == str:
            # read file
            try:
                with open(filename, 'br') as f:
                    self.data = buffer = f.read()
            except Exception as e:
                logger.error('%s open failed：%s' % (filename, str(e)))
                self.clear()
                return False

            if self.data == None:
                logger.error('%s load failed' % filename)
                self.clear()
                return False
        else:
            self.clear()
            return False

        if len(buffer) < 8:
            logger.error('%s load failed, file only %d bytes' %
                         (filename, len(buffer))
                         )
            self.clear()
            return False

        # index range
        index_begin = int4(buffer, 0)
        index_end = int4(buffer, 4)
        if index_begin > index_end or \
                (index_end - index_begin) % 7 != 0 or \
                index_end + 7 > len(buffer):
            logger.error('%s index error' % filename)
            self.clear()
            return False

        self.index_begin = index_begin
        self.index_end = index_end
        self.index_count = (index_end - index_begin) // 7 + 1

        if not loadindex:
            logger.info('%s %s bytes, %d segments. without index.' %
                        (filename, format(len(buffer), ','), self.index_count)
                        )
            self.__fun = self.__raw_search
            return True

        # load index
        self.idx1 = array.array('L')
        self.idx2 = array.array('L')
        self.idxo = array.array('L')

        try:
            for i in range(self.index_count):
                ip_begin = int4(buffer, index_begin + i * 7)
                offset = int3(buffer, index_begin + i * 7 + 4)

                # load ip_end
                ip_end = int4(buffer, offset)

                self.idx1.append(ip_begin)
                self.idx2.append(ip_end)
                self.idxo.append(offset + 4)
        except:
            logger.error('%s load index error' % filename)
            self.clear()
            return False

        logger.info('%s %s bytes, %d segments. with index.' %
                    (filename, format(len(buffer), ','), len(self.idx1))
                    )
        self.__fun = self.__index_search
        return True

    def __get_addr(self, offset):
        # mode 0x01, full jump
        mode = self.data[offset]
        if mode == 1:
            offset = int3(self.data, offset + 1)
            mode = self.data[offset]

        # country
        if mode == 2:
            off1 = int3(self.data, offset + 1)
            c = self.data[off1:self.data.index(b'\x00', off1)]
            offset += 4
        else:
            c = self.data[offset:self.data.index(b'\x00', offset)]
            offset += len(c) + 1

        # province
        if self.data[offset] == 2:
            offset = int3(self.data, offset + 1)
        p = self.data[offset:self.data.index(b'\x00', offset)]

        return c.decode('gb18030', errors='replace'), \
               p.decode('gb18030', errors='replace')

    def lookup(self, ip_str: str) -> Union[Tuple[str, str], None]:
        '''查找IP地址的归属地。
           找到则返回一个含有两个字符串的元组，如：('国家', '省份')
           没有找到结果，则返回一个None。'''
        ip = struct.unpack(">I", socket.inet_aton(ip_str.strip()))[0]

        try:
            return self.__fun(ip)
        except:
            if not self.is_loaded():
                logger.error('Error: qqwry.dat not loaded yet.')
            else:
                raise

    def __raw_search(self, ip):
        l = 0
        r = self.index_count

        while r - l > 1:
            m = (l + r) // 2
            offset = self.index_begin + m * 7
            new_ip = int4(self.data, offset)

            if ip < new_ip:
                r = m
            else:
                l = m

        offset = self.index_begin + 7 * l
        ip_begin = int4(self.data, offset)

        offset = int3(self.data, offset + 4)
        ip_end = int4(self.data, offset)

        if ip_begin <= ip <= ip_end:
            return self.__get_addr(offset + 4)
        else:
            return None

    def __index_search(self, ip):
        posi = bisect.bisect_right(self.idx1, ip) - 1

        if posi >= 0 and self.idx1[posi] <= ip <= self.idx2[posi]:
            return self.__get_addr(self.idxo[posi])
        else:
            return None

    def is_loaded(self) -> bool:
        '''是否已加载数据，返回True或False。'''
        return self.__fun != None

    def get_lastone(self) -> Union[Tuple[str, str], None]:
        '''返回最后一条数据，最后一条通常为数据的版本号。
           没有数据则返回一个None。
           如：('纯真网络', '2020年9月30日IP数据')'''
        try:
            offset = int3(self.data, self.index_end + 4)
            return self.__get_addr(offset + 4)
        except:
            return None

    def get_city(self, ip_str: str) -> Tuple[str, str] or None:
        info = self.lookup(ip_str)
        if info is None:
            return None
        if info[0][:2] in self.provinces[:6]:
            province = info[0][:2]
            city = province
        elif info[0][:2] in self.provinces[6:10]:
            province = info[0][:2]
            arr = info[0][2:].split('自治区', 1)
            if len(arr) == 2:
                city = arr[1].rstrip('市')
            else:
                city = info[0][2:].rstrip('市')
        elif info[0][:3] in self.provinces[10:12]:
            province = info[0][:3]
            city = info[0][3:].rstrip('市')
        elif info[0][:2] in self.provinces[12:]:
            province = info[0][:2]
            city = info[0][2:].lstrip('省').rstrip('市')
        else:
            return None
        return province, city


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        fn = 'qqwry.dat'
        q = QQwry()
        q.load_file(fn)

        for ipstr in sys.argv[1:]:
            s = q.lookup(ipstr)
            print('%s\n%s' % (ipstr, s))
    else:
        print('请以查询ip作为参数运行')
