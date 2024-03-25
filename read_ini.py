# https://qiita.com/mimitaro/items/3506a444f325c6f980b2
import configparser

# 基本の使い方
path = './data/config.ini'
config = configparser.ConfigParser()
config.read(path, encoding='utf-8')

print(config['DEFAULT']['User'])
print(config['DEFAULT']['Path'])
