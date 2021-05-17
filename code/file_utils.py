# -*- coding:utf-8 -*-
def readFromFileByLine(file, decode='utf-8'):
    result = []
    with open(file, 'r', encoding=decode) as f:
        for line in f:
            result.append(line.strip('\n'))
    return result

def writeFileFrom2StrPerLine(file, list):
    with open(file, 'a+', encoding='utf-8') as f:
        for item in list:
            f.write(item[0]+'\t'+item[1])
            f.write('\n')

def writeFileFrom1StrPerLine(file, list):
    with open(file, 'a+', encoding='utf-8') as f:
        for item in list:
            f.write(item)
            f.write('\n')

def writeFile(file, line,encode='utf-8'):
    with open(file, 'a+', encoding=encode) as f:
        f.write(line)
        f.write('\n')