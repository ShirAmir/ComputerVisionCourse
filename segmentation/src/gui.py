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
main_frame = ttk.Frame(root, padding="3 3 12 12")
main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

# define all the variables
image = []
train_img_path = StringVar()
labels_img_path = StringVar()
test_img_path = StringVar()
output_dir = StringVar()

# define teh button widgets
ttk.Button(main_frame, text="Train Image", command=get_train_img, width=15).grid(column=0, row=1, sticky=W)
ttk.Button(main_frame, text="Segment Image", command=get_segment_img, width=15).grid(column=0, row=2, sticky=W)
ttk.Button(main_frame, text="New Image", command=get_new_img, width=15).grid(column=0, row=3, sticky=W)
ttk.Button(main_frame, text="Output", command=get_output, width=15).grid(column=0, row=4, sticky=W)
ttk.Button(main_frame, text="GO!", command=compute, width=15).grid(column=0, row=4, sticky=W)

# define the labels for debugging
ttk.Label(main_frame, textvariable=train_img_path).grid(column=1, row=1, sticky=(W, E))
ttk.Label(main_frame, textvariable=labels_img_path).grid(column=1, row=2, sticky=(W, E))
ttk.Label(main_frame, textvariable=test_img_path).grid(column=1, row=3, sticky=(W, E))
ttk.Label(main_frame, textvariable=output_dir).grid(column=1, row=4, sticky=(W, E))

# change appearnace
for child in main_frame.winfo_children(): child.grid_configure(padx=5, pady=5)

# execute GUI application
root.mainloop()
