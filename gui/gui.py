from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from PIL import Image, ImageTk


class App(Tk):
    def __init__(self, appw, appy):
        super().__init__()

        self.title("Image segmentation")

        self.GRIDPADX = 15
        self.GRIDPADY = 15

        imgw = (appw // 3) - self.GRIDPADX
        self.IMGSIZE = (imgw, imgw)

        self.geometry(f'{appw}x{appy}+0+0')
        self.resizable(False, False)

        self.titleframe = Frame(self)
        self.buttonframe = Frame(self)
        self.photoframe = Frame(self)
        self.classframe = Frame(self)

        self.titleframe.grid(row=0, column=0, columnspan=4, sticky='nw')
        self.buttonframe.grid(row=1, column=0, columnspan=4, sticky='nw')
        self.photoframe.grid(row=2, column=0, columnspan=4, sticky='nw')
        self.classframe.grid(row=4, column=0, sticky='nw')

        Label(self.titleframe, text="Semantic segmentation", font=("Arial Bold", 25)) \
            .grid(row=0, column=0, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Choose image", command=self.select_file) \
            .grid(row=0, column=1, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Regenerate", command=self.rerun) \
            .grid(row=0, column=2, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Quit", command=self.destroy) \
            .grid(row=0, column=3, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        self.img1 = Label(self.photoframe)
        self.img2 = Label(self.photoframe)
        self.img3 = Label(self.photoframe)

        self.img1.grid(row=1, column=0, padx=2, pady=self.GRIDPADY, sticky='w')
        self.img2.grid(row=1, column=1, padx=2, pady=self.GRIDPADY, sticky='w')
        self.img3.grid(row=1, column=2, padx=2, pady=self.GRIDPADY, sticky='w')

        self.ctitle = Label(self, text="List of detected classes", font=("Arial Bold", 15))
        self.ctitle.grid_forget()

    def anchor_photo(self, tkobj, photo):
        tkobj['image'] = photo
        tkobj.photo = photo

    def rerun(self):
        pass

    def select_file(self):
        file = filedialog.askopenfilename(title="Select an image", filetypes=[("Image file", "*.jpg *.jpeg *.png")])

        if file == "" or file is None:
            return

        image = Image.open(file)
        image.thumbnail(self.IMGSIZE, Image.LANCZOS)

        photo = ImageTk.PhotoImage(image)

        self.anchor_photo(self.img1, photo)
        self.anchor_photo(self.img2, photo)
        self.anchor_photo(self.img3, photo)

        self.ctitle.grid(row=3, column=0, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        labels = []
        cnames = ["yo", "bro", "whatup", "haha"]

        for count, n in enumerate(cnames):
            labels.append(Label(self.classframe, text=n, font=("Arial Bold", 10)))
            labels[count].grid(row=0, column=count, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')
