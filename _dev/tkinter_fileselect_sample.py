import os
import sys
import shutil
import traceback
import threading
from datetime import datetime 
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import pytz
# from utils.logger import get_logger
# from utils.common import DualOutput

class ScoringExecutor:
    def __init__(self, true_path, pred_path):
        self.true_path = true_path
        self.pred_path = pred_path

    def __call__(self):
        import time
        time.sleep(5)
            

class MainGui(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        
        self.item_paths = None
        self.acc_list = []
        
        iconfile = "template/graphmagnifier_117810.ico"
        self.master.iconbitmap(default=iconfile)
        self.master.geometry("600x180")
        self.master.title("CIS-AVI-MetricLearning ScoringApp Ver.0.0.0")
        self.pack()
        
        # 正解フォルダ選択ボタン
        self.truefolder_label = tk.Label(self, text="true label folder:")
        self.truefolder_label.grid(row=0, column=0, padx=10, pady=10)
        self.truefolder_entry = tk.Entry(self, width=40)
        self.truefolder_entry.grid(row=0, column=1, padx=10, pady=10)
        self.truefolder_button = tk.Button(self, text="select folder", command=self.select_truefolder)
        self.truefolder_button.grid(row=0, column=2, padx=10, pady=10)

        # 予測フォルダ選択ボタン
        self.predfolder_label = tk.Label(self, text="pred csv folder:")
        self.predfolder_label.grid(row=1, column=0, padx=10, pady=10)
        self.predfolder_entry = tk.Entry(self, width=40)
        self.predfolder_entry.grid(row=1, column=1, padx=10, pady=10)
        self.predfolder_button = tk.Button(self, text="select folder", command=self.select_predfolder)
        self.predfolder_button.grid(row=1, column=2, padx=10, pady=10)

        
        # 実行ボタン
        self.run_button = tk.Button(self, text="Scoring", command=self.main_process)
        self.run_button.grid(row=5, column=1, padx=10, pady=10)
        
    def select_truefolder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.truefolder_entry.delete(0, tk.END)
            self.truefolder_entry.insert(0, folder_path)

    def select_predfolder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.predfolder_entry.delete(0, tk.END)
            self.predfolder_entry.insert(0, folder_path)
    
    def show_progress_popup(self):
        progress_popup = tk.Toplevel(self)
        progress_popup.title("評価中")
        progress_popup.geometry("300x100")
        label = tk.Label(progress_popup, text=f"評価中です。\nお待ちください...")
        label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_popup, mode="indeterminate")
        progress_bar.pack(pady=10, padx=10, fill=tk.X)
        progress_bar.start()

        return progress_popup, progress_bar
    
    def check_thread(self,thread, progress_popup, progress_bar):
        if thread.is_alive():
            self.after(100, self.check_thread,thread, progress_popup, progress_bar)
        else:
            progress_bar.stop()
            progress_popup.destroy()
            messagebox.showinfo("完了", "評価が完了しました")
            

    def main_process(self):
        true_path = self.truefolder_entry.get()
        pred_path = self.predfolder_entry.get()
        if true_path and pred_path:
            # 学習開始
            print(f"\nscoring start:***********************")
            scoring_executor = ScoringExecutor(true_path,pred_path) 
            scoring_thread = threading.Thread(target=scoring_executor)
            scoring_thread.start()
            # プログレスバーの表示
            progress_popup, progress_bar = self.show_progress_popup()
            self.after(100, self.check_thread, scoring_thread, progress_popup, progress_bar)
                
        else:
            messagebox.showerror("エラー", "フォルダを選んでください")
            return 1
        return 0

class ErrorCatchTk(tk.Tk):
    # # エラーキャッチ用
    # def report_callback_exception(self, exc, val, tb):
    #     lg.error("Exception in Tkinter callback", exc_info=(exc, val, tb)) 
    def dummyfunc(self):
        pass


def create_dir(folder_path, delete=False):
    os.makedirs(folder_path, exist_ok=True)
    if delete:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


if __name__ == "__main__":
    # # logフォルダの作成
    # create_dir("log")
    # # 標準出力の保存設定
    # original_stdout = sys.stdout
    # sys.stdout = DualOutput(f'log/stdout_log.txt', original_stdout)
    
    # エラーログの設定
    # lg = get_logger(__name__,'log/error_log.txt')
    # timezone = pytz.timezone("Asia/Tokyo")
    # now = datetime.now(timezone)
    # lg.debug(f'{now} logging start')
    

    root = ErrorCatchTk()
    app = MainGui(master=root)
    app.mainloop()
    # try:
    #     root = ErrorCatchTk()
    #     app = MainGui(master=root)
    #     app.mainloop()
    # except:
    #     lg.error(traceback.format_exc())