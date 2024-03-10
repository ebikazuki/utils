# https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0   
import argparse    

parser = argparse.ArgumentParser(description='このプログラムの説明')    

parser.add_argument('arg1', type=str, help='arg1の説明')
parser.add_argument('arg2', help='arg2の説明')
parser.add_argument('--arg3', default=1, help='arg3の説明')    # オプション引数
parser.add_argument('-a', '--arg4')   # 省略形

args = parser.parse_args()

print('arg1=',args.arg1, 'type=', type(args.arg1))
print('arg2=',args.arg2)
if args.arg3 is not None:
    print('arg3=',args.arg3)
if args.arg4 is not None:
    print('arg4=',args.arg4)
