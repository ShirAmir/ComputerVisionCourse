from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

# define callback functions
def get_train_img():
    train_img_path.set(askopenfilename())
    
def get_segment_img():
    labels_img_path.set(askopenfilename())

def get_new_img():
    test_img_path.set(askopenfilename())

def get_output():
    output_dir.set(askdirectory())

def compute():
    # the segmentation implementation goes here
    return image

# define root and main-frame of the GUI
root = Tk()
root.title("Segmentation")
frame = ttk.Frame(root)
frame.grid(column=0, row=0, sticky=(N, W, E, S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

# define all the variables
image = []
train_img_path = StringVar()
labels_img_path = StringVar()
test_img_path = StringVar()
output_dir = StringVar()

# define teh button widgets
train_img_btn = ttk.Button(frame, text="Train Image", command=get_train_img, width=15)
train_img_btn.grid(column=0, row=1, sticky=W)
segment_img_btn = ttk.Button(frame, text="Segment Image", command=get_segment_img, width=15)
segment_img_btn.grid(column=0, row=2, sticky=W)
test_img_btn = ttk.Button(frame, text="New Image", command=get_new_img, width=15)
test_img_btn.grid(column=0, row=3, sticky=W)
output_path_btn = ttk.Button(frame, text="Output", command=get_output, width=15)
output_path_btn.grid(column=0, row=4, sticky=W)
run_btn = ttk.Button(frame, text="GO!", command=compute, width=15)
run_btn.grid(column=0, row=5, sticky=W)

# define the labels for debugging
ttk.Label(frame, textvariable=train_img_path).grid(column=1, row=1, sticky=(W, E))
ttk.Label(frame, textvariable=labels_img_path).grid(column=1, row=2, sticky=(W, E))
ttk.Label(frame, textvariable=test_img_path).grid(column=1, row=3, sticky=(W, E))
ttk.Label(frame, textvariable=output_dir).grid(column=1, row=4, sticky=(W, E))

# change appearnace
for child in frame.winfo_children(): child.grid_configure(padx=5, pady=5)

# execute GUI application
root.mainloop()
