from Crypto.Cipher import DES


def decrypt(key: bytes, s: str):
    d = DES.new(key, DES.MODE_ECB)
    dd = d.decrypt(bytes.fromhex(s))
    pad_num = dd[-1]
    res_str = dd[:-pad_num].decode('utf-8')
    return res_str


def encrypt(key: bytes, s: str):
    d = DES.new(key, DES.MODE_ECB)
    s = s.encode('utf-8')
    pad_num = 8 - len(s) % 8
    b = s + bytes([pad_num] * pad_num)
    return bytes.hex(d.encrypt(b))
