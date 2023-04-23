from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

APPW = 1400
APPY = 600
GRIDPADX = 15
GRIDPADY = 15
IMGSIZE = (350, 350)

window = Tk()
window.title("Image segmentation")

window.geometry(f'{APPW}x{APPY}+0+0')
window.resizable(False, False)

window.config(bg="lightgrey")

titleframe = Frame(window, bg="lightgrey")
buttonframe = Frame(window, bg="lightgrey")
photoframe = Frame(window, bg="lightgrey")
classframe = Frame(window, bg="lightgrey")

titleframe.grid(row=0, column=0, columnspan=4, sticky='n')
buttonframe.grid(row=1, column=0, columnspan=4, sticky='n')
photoframe.grid(row=2, column=0, columnspan=4, sticky='n')
classframe.grid(row=3, column=0, columnspan=4, sticky='n')


def rerun():
    pass


def select_file():
    file = filedialog.askopenfilename(title="Select an image", filetypes=[("Image file", "*.jpg *.jpeg *.png")])
    image = Image.open(file)
    image.thumbnail(IMGSIZE, Image.LANCZOS)

    photo = ImageTk.PhotoImage(image)

    lb1 = Label(photoframe, image=photo)
    lb1.photo = photo
    lb1.grid(row=1, column=0, padx=GRIDPADX, pady=GRIDPADY)

    lb2 = Label(photoframe, image=photo)
    lb2.photo = photo
    lb2.grid(row=1, column=1, padx=GRIDPADX, pady=GRIDPADY)

    lb3 = Label(photoframe, image=photo)
    lb3.photo = photo
    lb3.grid(row=1, column=2, padx=GRIDPADX, pady=GRIDPADY)

    lbc = Label(classframe, text="yo")
    lbc.grid(row=0, column=0, padx=GRIDPADX, pady=GRIDPADY)


maintitle = Label(titleframe, text="Semantic segmentation", bg="lightgrey", font=("Arial Bold", 25))
filebtn = Button(buttonframe, text="Choose image", command=select_file, padx=10, pady=10)
regenbtn = Button(buttonframe, text="Regenerate", command=rerun, padx=10, pady=10)
quitbtn = Button(buttonframe, text="Quit", command=window.destroy, padx=10, pady=10)

maintitle.grid(row=0, column=0, padx=GRIDPADX, pady=GRIDPADY)
filebtn.grid(row=0, column=1, padx=GRIDPADX, pady=GRIDPADY)
regenbtn.grid(row=0, column=2, padx=GRIDPADX, pady=GRIDPADY)
quitbtn.grid(row=0, column=3, padx=GRIDPADX, pady=GRIDPADY)

window.mainloop()
