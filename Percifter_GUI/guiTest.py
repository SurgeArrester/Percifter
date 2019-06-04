import tkinter
from tkinter import filedialog
import os

window = tkinter.Tk()
window.title("File Preview")

text = tkinter.Text(window, height = 30, width = 30)
text.pack(side = "bottom")


pathEnter = tkinter.Entry(window)
pathEnter.pack(side = "left")


get = tkinter.Button(window, text = "OK", command = lambda: open_file(pathEnter.get())).pack(side = "right")

folder = tkinter.Button(window, text = "Folder", command = lambda: open_folder()).pack(side = "right")

def open_file(path):
	with open(path, 'r') as f:
		text.insert(tkinter.END, f.read())

def open_folder():
    file_select =  filedialog.askopenfilename(initialdir = "/Desktop",title = "Select file",filetypes = (("rtf files",".rtf"),("all files",".*")))
    pathEnter.insert(0, file_select)
    open_file(file_select)
    execution_string = running_function(filepath_from_selection)
    os.popen(execution_string)

window.mainloop()

def running_function(file_path):
    return f"python /path/to/Percifter/PersistenceGenerator.py -i" + str(file_path) + "> out.txt"
