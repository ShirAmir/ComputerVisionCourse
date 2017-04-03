from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

def get_train():
    train_img_path.set(askopenfilename())
    
def get_segment():
    labels_img_path.set(askopenfilename())

def get_new():
    test_img_path.set(askopenfilename())

def get_output():
    output_dir.set(askdirectory())

def compute():
    # the segmentation implementation goes here
    return image

root = Tk()
root.title("Segmentation")
root.attributes("-fullscreen", True)
root.configure(background='midnight blue')
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.overrideredirect(1)
root.geometry("%dx%d+0+0" % (w, h))

root.focus_set() # <-- move focus to this widget
root.bind("<Escape>", lambda e: e.widget.quit())

# define all the variables
image = []
train_img_path = StringVar()
labels_img_path = StringVar()
test_img_path = StringVar()
output_dir = StringVar()

headline = Label(root, font=(None, 20), text="Segmentation App", anchor=CENTER).grid(row=0, column=0, columnspan=7)

btn1 = Button(root, command=get_train, text="Train Image", width=15)
btn1.grid(row=1, column=0, padx = 20, pady = 10)
btn2 = Button(root, command=get_segment, text="Labeled Image", width=15)
btn2.grid(row=2, column=0, padx = 20, pady = 10)
btn3 = Button(root, command=get_new, text="Test Image", width=15)
btn3.grid(row=3, column=0, padx = 20, pady = 10)
btn4 = Button(root, command=get_output, text="Output Path", width=15)
btn4.grid(row=4, column=0, padx = 20, pady = 10)
btn5 = Button(root, command=compute, text="GO!", width=15)
btn5.grid(row=5, column=0, padx = 20, pady = 10)

image_frame = Frame(bg="black", width=0.75*w, height=0.75*h).grid(row=1, column=1, columnspan=5, rowspan=20, sticky=E)

btn1 = Button(root, command=get_train, text="Train Image", width=15)
btn1.grid(row=1, column=6, padx = 20, pady = 10)
btn2 = Button(root, command=get_segment, text="Labeled Image", width=15)
btn2.grid(row=2, column=6, padx = 20, pady = 10)
btn3 = Button(root, command=get_new, text="Test Image", width=15)
btn3.grid(row=3, column=6, padx = 20, pady = 10)
btn4 = Button(root, command=get_output, text="Output Path", width=15)
btn4.grid(row=4, column=6, padx = 20, pady = 10)
btn5 = Button(root, command=compute, text="GO!", width=15)
btn5.grid(row=5, column=6, padx = 20, pady = 10)

# Label(root, textvariable=train_img_path).grid(column=1, row=0) 
# Label(root, textvariable=labels_img_path).grid(column=1, row=1)
# Label(root, textvariable=test_img_path).grid(column=1, row=2)
# Label(root, textvariable=output_dir).grid(column=1, row=3)

root.mainloop()