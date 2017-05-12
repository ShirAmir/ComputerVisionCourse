# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import cv2
import train
import test

def do_train():
    """ callback function """
    train.run_training()
    training_msg.set("Training Complete!")

def do_test():
    """ callback function """
    training_msg.set("")
    res_path = test.run_testing(test_img_path.get(), output_dir.get(), float(thresh.get()))
    """res = PIL.Image.open(res_path)
    res = PIL.ImageTk.PhotoImage(res)
    # image = Label(canvas, image=res, anchor=CENTER)
    image.configure(image=res)
    image.image = res
    image.grid(row=2, column=2, columnspan=3, rowspan=12)
    """

def add_training_set():
    """ callback function """
    training_msg.set("")
    new_training_set_path.set(askdirectory())
    train.add_training_set(new_training_set_path.get())

def set_image_path():
    """ callback function """
    training_msg.set("")
    test_img_path.set(askopenfilename())

def set_output_dir():
    """ callback function """
    training_msg.set("")
    output_dir.set(askdirectory())

root = Tk()
root.title("Facial Recognition")
root.configure(background='#ccffcc')
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

root.focus_set() # move focus to this widget
root.bind("<Escape>", lambda e: e.widget.quit())

# Define all the variables
new_training_set_path = StringVar()
test_img_path = StringVar()
output_dir = StringVar()
thresh = StringVar()
training_msg = StringVar()

# Header
headline = Label(root, font="Gisha 20 bold", bg='#ccffcc', fg='#006600', text="Facial Recognition App", anchor=CENTER)
headline.grid(row=0, column=0, columnspan=7, padx=10, pady=10)

# Left input and output settings
btn1 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=do_train, text="Train", width=12)
btn1.grid(row=1, column=0, padx=15, pady=0)
btn2 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=do_test, text="Test", width=12)
btn2.grid(row=2, column=0, padx=15, pady=0)
btn3 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=add_training_set, text="Add Training \n Set", width=12)
btn3.grid(row=3, column=0, padx=15, pady=0)
btn4 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=set_image_path, text="Set Tested \n Image Path", width=12)
btn4.grid(row=4, column=0, padx=15, pady=0)
test_img_path.set("../images/test_data_sets/Lleyton_Hewitt/Lleyton_Hewitt_0024.jpg")
btn5 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=set_output_dir, text="Set Output \n Directory", width=12)
btn5.grid(row=5, column=0, padx=15, pady=0)
output_dir.set("../results/")

# Main frame for showing results
canvas = Canvas(relief=RIDGE, bd=5, width=0.75*w, height=0.75*h).grid(row=1, column=1, columnspan=5, rowspan=22, sticky=E)
image = Label(canvas, image="", anchor=CENTER)

# Right parameter settings
label1 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Classification \n Threshold:", width=15)
label1.grid(row=1, column=6, padx=0, pady=0)
entry1 = Entry(root, font="Gisha 12", fg='#006600', textvariable=thresh, width=12)
entry1.grid(row=2, column=6, padx=15, pady=0)
thresh.set("9")

# Footer inside images frame
training_msg.set("")
msg_banner = Label(root, font="Gisha 14", fg='#006600', bg='#ccffcc', textvariable=training_msg, anchor=CENTER)
msg_banner.grid(row=10, column=0, columnspan=7, padx=10, pady=10)
explanations = "Press each button once and be patient :) \n press Esc to leave"
footer = Label(root, font="Gisha 14", fg='#006600', bg='#ccffcc', text=explanations, anchor=CENTER)
footer.grid(row=20, column=0, columnspan=7, padx=10, pady=10)

root.mainloop()