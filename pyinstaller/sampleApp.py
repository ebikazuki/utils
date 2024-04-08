from tkinter import *

root = Tk()                                             # Tkinter Window
root.title("sample_v7.py")                              # Window Title
root.geometry("400x300")                                # Window Size - Width / Height

tk_button_1 = Button(root, text="button 1")             # Button Initilization
tk_button_2 = Button(root, text="button 2")             # Button Initilization
tk_button_3 = Button(root, text="button 3")             # Button Initilization
tk_button_4 = Button(root, text="button 4")             # Button Initilization

tk_button_1.place(x=50,  y=50,  width=50, height=50)    # Button Configuration - X / Y / Width / Padding / Height
tk_button_2.place(x=150, y=50,  width=50, height=50)    # Button Configuration - X / Y / Width / Padding / Height
tk_button_3.place(x=50,  y=150, width=50, height=50)    # Button Configuration - X / Y / Width / Padding / Height
tk_button_4.place(x=150, y=150, width=50, height=50)    # Button Configuration - X / Y / Width / Padding / Height

root.mainloop()     