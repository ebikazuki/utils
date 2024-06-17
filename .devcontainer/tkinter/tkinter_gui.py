import tkinter as tk
from tkinter import filedialog, messagebox

class MainGui(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master.title("CIS-GoodRaw 切出しアプリ Ver.0.0")
        self.pack()
        
        # ファイル選択ボタンとテキストボックス
        self.file_label = tk.Label(self, text="ファイル:")
        self.file_label.grid(row=0, column=0, padx=10, pady=10)
        
        self.file_entry = tk.Entry(self, width=40)
        self.file_entry.grid(row=0, column=1, padx=10, pady=10)
        
        self.file_button = tk.Button(self, text="ファイル選択", command=self.select_file)
        self.file_button.grid(row=0, column=2, padx=10, pady=10)
        
        # 数値条件入力用テキストボックス
        self.condition1_label = tk.Label(self, text="数値条件1:")
        self.condition1_label.grid(row=1, column=0, padx=10, pady=10)
        
        self.condition1_entry = tk.Entry(self)
        self.condition1_entry.grid(row=1, column=1, padx=10, pady=10)
        
        self.condition2_label = tk.Label(self, text="数値条件2:")
        self.condition2_label.grid(row=2, column=0, padx=10, pady=10)
        
        self.condition2_entry = tk.Entry(self)
        self.condition2_entry.grid(row=2, column=1, padx=10, pady=10)
        
        self.condition3_label = tk.Label(self, text="数値条件3:")
        self.condition3_label.grid(row=3, column=0, padx=10, pady=10)
        
        self.condition3_entry = tk.Entry(self)
        self.condition3_entry.grid(row=3, column=1, padx=10, pady=10)
        
        # 実行ボタン
        self.run_button = tk.Button(self, text="処理A実行", command=self.run_process_a)
        self.run_button.grid(row=4, column=1, padx=10, pady=10)
        
    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
    
    def run_process_a(self):
        try:
            condition1 = int(self.condition1_entry.get())
            condition2 = int(self.condition2_entry.get())
            condition3 = int(self.condition3_entry.get())
            
            # ここに処理Aの実装を追加
            messagebox.showinfo("実行", f"処理Aを実行しました。\n条件1: {condition1}\n条件2: {condition2}\n条件3: {condition3}")
        except ValueError:
            messagebox.showerror("エラー", "数値条件には整数を入力してください。")
    
app = MainGui()
app.mainloop()
