from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import segment #our segmentation module

def get_train():
    train_img_path.set(askopenfilename())
    
def get_segment():
    labels_img_path.set(askopenfilename())

def get_new():
    test_img_path.set(askopenfilename())

def get_output():
    output_dir.set(askdirectory())

def compute():
    # Prepare parameters
    kwargs = {}
    kwargs['TRAIN_IMG_PATH'] = train_img_path.get()
    kwargs['LABELS_IMG_PATH'] = labels_img_path.get()
    kwargs['TEST_IMG_PATH'] = test_img_path.get()
    kwargs['OUTPUT_DIR'] = output_dir.get()
    kwargs['FRAG_AMOUNT'] = frag_amount.get()
    kwargs['PATCH_SIZE'] = patch_size.get()
    kwargs['GRABCUT_THRESH'] = grabcut_thresh.get()
    kwargs['GRABCUT_ITER'] = grabcut_iter.get()
    kwargs['SLIC_SIGMA'] = slic_sigma.get()

    # Compute segmentation
    segment.segment_image(**kwargs)

root = Tk()
root.title("Segmentation")
root.attributes("-fullscreen", True)
root.configure(background='#ccffcc')
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.overrideredirect(1)
root.geometry("%dx%d+0+0" % (w, h))

root.focus_set() # <-- move focus to this widget
root.bind("<Escape>", lambda e: e.widget.quit())

# Define all the variables
image = []
train_img_path = StringVar()
labels_img_path = StringVar()
test_img_path = StringVar()
output_dir = StringVar()
frag_amount = StringVar()
patch_size = StringVar()
grabcut_thresh = StringVar()
grabcut_iter = StringVar()
slic_sigma = StringVar()

# Header
headline = Label(root, font="Gisha 20 bold", bg='#ccffcc', fg='#006600', text="Segmentation App", anchor=CENTER)
headline.grid(row=0, column=0, columnspan=7, padx=10, pady=10)

# Left input and output settings
btn1 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_train, text="Train Image", width=12)
btn1.grid(row=1, column=0, padx=15, pady=0)
btn2 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_segment, text="Labeled Image", width=12)
btn2.grid(row=2, column=0, padx=15, pady=0)
btn3 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_new, text="Test Image", width=12)
btn3.grid(row=3, column=0, padx=15, pady=0)
btn4 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_output, text="Output Path", width=12)
btn4.grid(row=4, column=0, padx=15, pady=0)
btn5 = Button(root, font="Gisha 12 bold", fg='#006600', bg='#b3ffb3', command=compute, text="GO!", width=11)
btn5.grid(row=5, column=0, padx=15, pady=0)

# Main frame for showing results
image_frame = Frame(relief=RIDGE, bd=5, width=0.75*w, height=0.75*h).grid(row=1, column=1, columnspan=5, rowspan=22, sticky=E)

image = Label(image_frame, text="nanabanana", anchor=CENTER)
image.grid(row=6, column=3)

# Right parameter settings
label1 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Amount \n of Fragments:", width=15)
label1.grid(row=1, column=6, padx=0, pady=0)
entry1 = Entry(root, font="Gisha 12", fg='#006600', textvariable=frag_amount, width=12)
entry1.grid(row=2, column=6, padx=15, pady=0)
frag_amount.set("100")

label2 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Patch Size:", width=12)
label2.grid(row=3, column=6, padx=15, pady=0)
entry2 = Entry(root, font="Gisha 12", fg='#006600', textvariable=patch_size, width=12)
entry2.grid(row=4, column=6, padx=15, pady=0)
patch_size.set("9")

label3 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Grabcut \n Threshold:", width=12)
label3.grid(row=5, column=6, padx=0, pady=0)
entry3 = Entry(root, font="Gisha 12", fg='#006600', textvariable=grabcut_thresh, width=12)
entry3.grid(row=6, column=6, padx=15, pady=0)
grabcut_thresh.set("0.01")

label4 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Grabcut \n Iterations:", width=12)
label4.grid(row=7, column=6, padx=0, pady=0)
entry4 = Entry(root, font="Gisha 12", fg='#006600', textvariable=grabcut_iter, width=12)
entry4.grid(row=8, column=6, padx=15, pady=0)
grabcut_iter.set("5")

label5 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="SLIC Sigma:", width=12)
label5.grid(row=9, column=6, padx=0, pady=0)
entry5 = Entry(root, font="Gisha 12", fg='#006600', textvariable=slic_sigma, width=12)
entry5.grid(row=10, column=6, padx=15, pady=0)
slic_sigma.set("5")

footer = Label(root, font="Gisha 14", fg='#006600', bg='#ccffcc', text="Press Esc to leave", anchor=CENTER)
footer.grid(row=20, column=0, columnspan=7, padx=10, pady=10)

# Label(root, textvariable=train_img_path).grid(column=1, row=0) 
# Label(root, textvariable=labels_img_path).grid(column=1, row=1)
# Label(root, textvariable=test_img_path).grid(column=1, row=2)
# Label(root, textvariable=output_dir).grid(column=1, row=3)

root.mainloop()