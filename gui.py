import time
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

import torch
import torchvision.transforms as T
from PIL import Image, ImageTk

import models
from imgutils import segment_map
from tools import get_classes


class App(Tk):
    def __init__(self, appw, appy):
        super().__init__()

        self.title("Image segmentation")

        self.GRIDPADX = 15
        self.GRIDPADY = 15

        imgw = (appw // 4) - self.GRIDPADX
        self.IMGSIZE = (imgw, imgw)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cats = get_classes("pascal.txt")
        self.nb_classes = len(self.cats)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(self.nb_classes)])[:, None] * palette
        self.colormap = (colors % 255).numpy().astype("uint8")

        self.transform = T.Compose([
            T.Resize(self.IMGSIZE),
            T.CenterCrop(self.IMGSIZE),
            T.ToTensor()
        ])

        self.model = models.load_model("dlab")[0].eval()
        self.inf_img = None

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

        Button(self.buttonframe, text="Regenerate", command=self.inference) \
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

    def inference(self):

        if self.inf_img is not None:
            with torch.no_grad():
                inp = self.inf_img.unsqueeze(0).to(self.device)
                inp = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inp)

                st = time.time()
                out = self.model.to(self.device)(inp)['out']
                end = time.time()

            print(f"Inference took: {end - st:.2f}", )

            f_img = self.inf_img.permute(1, 2, 0)
            seg, overlay, cnames = segment_map(out, f_img, self.colormap, self.cats, self.nb_classes)

            return seg, overlay, cnames

    def select_file(self):
        file = filedialog.askopenfilename(title="Select an image", filetypes=[("Image file", "*.jpg *.jpeg *.png")])

        if file == "" or file is None:
            return

        image = Image.open(file)
        image.thumbnail(self.IMGSIZE, Image.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        self.inf_img = self.transform(image)

        seg, over, cnames = self.inference()

        seg = Image.fromarray(seg)
        over = Image.fromarray(over)

        seg.thumbnail(self.IMGSIZE, Image.LANCZOS)
        over.thumbnail(self.IMGSIZE, Image.LANCZOS)

        seg = ImageTk.PhotoImage(seg)
        over = ImageTk.PhotoImage(over)

        self.anchor_photo(self.img1, photo)
        self.anchor_photo(self.img2, seg)
        self.anchor_photo(self.img3, over)

        self.ctitle.grid(row=3, column=0, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        labels = []

        for count, n in enumerate(cnames):
            labels.append(Label(self.classframe, text=n, font=("Arial Bold", 10)))
            labels[count].grid(row=0, column=count, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')


app = App(appw=1400, appy=750)
app.mainloop()
