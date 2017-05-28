# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import cv2
import tracker

def load_video():
    """ callback function """
    video_path.set(askopenfilename())

def set_output_dir():
    """ callback function """
    output_dir.set(askdirectory())

def start():
    tracker.track(video_path.get(), int(contour_size_thresh.get()))

root = Tk()
root.title("Multi Tracker")
root.configure(background='#ccffcc')
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

root.focus_set() # move focus to this widget
root.bind("<Escape>", lambda e: e.widget.quit())

# Define all the variables
video_path = StringVar()
output_dir = StringVar()
contour_size_thresh = StringVar()

# Header
headline = Label(root, font="Gisha 20 bold", bg='#ccffcc', fg='#006600', text="Multi Tracker App", anchor=CENTER)
headline.grid(row=0, column=0, columnspan=7, padx=10, pady=10)

# Left input and output settings
btn1 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=load_video, text="Load Video", width=12)
btn1.grid(row=1, column=0, padx=15, pady=0)
video_path.set("../videos/bugs14.mp4")
btn2 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=set_output_dir, text="Set Output \n Directory", width=12)
btn2.grid(row=2, column=0, padx=15, pady=0)
output_dir.set("../results/")
btn3 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=start, text="Start", width=12)
btn3.grid(row=3, column=0, padx=15, pady=0)
label4 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Press b for \n bounding box", width=15)
label4.grid(row=4, column=0, padx=0, pady=0)
label5 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Press t for trail", width=15)
label5.grid(row=5, column=0, padx=0, pady=0)
label6 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Press d for distance", width=15)
label6.grid(row=6, column=0, padx=0, pady=0)
label7 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Press space to \n pause/play", width=15)
label7.grid(row=7, column=0, padx=0, pady=0)
label8 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Press Esc to quit", width=15)
label8.grid(row=8, column=0, padx=0, pady=0)

# Main frame for showing results
canvas = Canvas(relief=RIDGE, bd=5, width=0.75*w, height=0.75*h).grid(row=1, column=1, columnspan=5, rowspan=22, sticky=E)
image = Label(canvas, image="", anchor=CENTER)

# Right parameter settings
label1 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Contour Size:", width=15)
label1.grid(row=1, column=6, padx=0, pady=0)
entry1 = Entry(root, font="Gisha 12", fg='#006600', textvariable=contour_size_thresh, width=12)
entry1.grid(row=2, column=6, padx=15, pady=0)
contour_size_thresh.set("100")

# Footer inside images frame
explanations = "Press each button once and be patient :) \n press Esc to leave"
footer = Label(root, font="Gisha 14", fg='#006600', bg='#ccffcc', text=explanations, anchor=CENTER)
footer.grid(row=20, column=0, columnspan=7, padx=10, pady=10)

root.mainloop()