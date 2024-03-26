from concurrent.futures import ProcessPoolExecutor
from shutil import copy
import os

def fn(i):
    src = f"from/{i}.png"
    dst = f"to{i}/{i}.png"
    return copy(src, dst)

def main():   
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # -----(2)
        for result in executor.map(fn, range(4)):
            print(result)
            

if __name__ == '__main__':
    main()
    
    
# def fn(idx, d):  # -------------------(1)
#     # for文の1つ単位の処理を関数化する
#     time.sleep(0.1)
#     return idx, d


# def fn2(d):  # -------------------(1)
#     # for文の1つ単位の処理を関数化する
#     time.sleep(0.1)
#     return d


# def main():
#     data = list(range(1000))

#     # tqdmで経過が知りたい時
#     with tqdm(total=len(data)) as progress:
#         # 1. 引数にiterできないオブジェクトがある時
#         with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # -----(2)
#             futures = []  # 処理結果を保存するlist
#             for i, d in enumerate(data):  # -------(3)
#                 future = executor.submit(fn, i, d)
#                 future.add_done_callback(lambda p: progress.update()) # tqdmで経過が知りたい時
#                 futures.append(future)
#             result = [f.result() for f in futures]

#     # 2. 引数がiterできる場合
#     with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # -----(2)
#         result = list(tqdm(executor.map(fn2, data), total=len(data)))


# if __name__ == "__main__":
#     main()