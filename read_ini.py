# https://qiita.com/mimitaro/items/3506a444f325c6f980b2
import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

var1 = config['DEFAULT']['User']
path = config['DEFAULT']['Path']

print('var1 :', var1)
print('path :', path)


