import time
import os
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

import torch
import torchvision.transforms as T
from PIL import Image, ImageTk

import models
from cocodata import get_data
from imgutils import segment_map
from tools import get_classes


class App(Tk):
    def __init__(self, appw, appy):
        super().__init__()

        self.inf_img = None
        self.model = None
        self.transform = None
        self.model_choice = None
        self.labels = []

        self.title("Image segmentation")

        self.GRIDPADX = 15
        self.GRIDPADY = 15

        imgw = int(appw // 3) - self.GRIDPADX
        self.IMGSIZE = (imgw, imgw)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        _, _, self.test_ds, self.supcats = get_data(input_size=self.IMGSIZE, batch_size=None, sup=True, gui=True)
        self.pascal_cats = get_classes("pascal.txt")
        self.cats = self.pascal_cats
        self.nb_classes = len(self.cats)

        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(self.nb_classes)])[:, None] * self.palette
        self.colormap = (colors % 255).numpy().astype("uint8")

        self.load_premade()

        self.geometry(f'{appw}x{appy}+0+0')
        self.resizable(False, False)

        self.titleframe = Frame(self)
        self.buttonframe = Frame(self)
        self.photoframe = Frame(self)
        self.classframe = Frame(self)

        self.img1 = Label(self.photoframe)
        self.img2 = Label(self.photoframe)
        self.img3 = Label(self.photoframe)

        self.load_mainbtn()

        self.ctitle = Label(self, text="List of detected classes", font=("Arial Bold", 15))

        self.place_grid()

        self.clear_classes()

    def load_mainbtn(self):
        Label(self.titleframe, text="Semantic segmentation", font=("Arial Bold", 25)) \
            .grid(row=0, column=0, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Choose image", command=self.select_file) \
            .grid(row=0, column=1, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Generate", command=self.generate) \
            .grid(row=0, column=2, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Premade model", command=self.load_pm) \
            .grid(row=0, column=3, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Fine-tuned model", command=self.load_custom) \
            .grid(row=0, column=4, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        Button(self.buttonframe, text="Quit", command=self.destroy) \
            .grid(row=0, column=5, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

    def turnon_img(self):
        self.img1.grid(row=1, column=0, padx=2, pady=self.GRIDPADY, sticky='w')
        self.img2.grid(row=1, column=1, padx=2, pady=self.GRIDPADY, sticky='w')
        self.img3.grid(row=1, column=2, padx=2, pady=self.GRIDPADY, sticky='w')

    def place_grid(self):

        self.titleframe.grid(row=0, column=0, columnspan=6, sticky='nw')
        self.buttonframe.grid(row=1, column=0, columnspan=6, sticky='nw')
        self.photoframe.grid(row=2, column=0, columnspan=6, sticky='nw')
        self.classframe.grid(row=4, column=0, sticky='nw')

        self.ctitle.grid(row=3, column=0, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

        self.turnon_img()

    def clear_classes(self, cl_img=False):

        self.ctitle.grid_forget()
        self.classframe.grid_forget()

        for lab in self.labels:
            lab.destroy()

        self.labels = []

        if cl_img:
            self.img1.grid_forget()
            self.img2.grid_forget()
            self.img3.grid_forget()

    def load_pm(self):
        self.load_premade()
        self.clear_classes(cl_img=True)

    def load_premade(self):

        self.transform = T.Compose([
            T.Resize(self.IMGSIZE),
            T.CenterCrop(self.IMGSIZE),
            T.ToTensor()
        ])

        self.model_choice = "dlab"
        self.model = models.load_model(self.model_choice)[0].eval()

    def load_custom(self):
        path = f"pytorch_models/{self.model_choice}/{self.model_choice}_ft.pt"

        self.clear_classes(cl_img=True)

        if os.path.exists(path):
            print("Model file found, using pretrained model for inference\n")
            self.nb_classes = len(self.supcats)

            colors = torch.as_tensor([i for i in range(self.nb_classes)])[:, None] * self.palette
            self.colormap = (colors % 255).numpy().astype("uint8")

            self.model = torch.load(path)

    def anchor_photo(self, tkobj, photo):
        tkobj['image'] = photo
        tkobj.photo = photo

    def load_results(self, photo, seg, over, cnames):

        self.clear_classes()
        self.turnon_img()

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
        self.classframe.grid(row=4, column=0, sticky='nw')

        for count, n in enumerate(cnames):
            self.labels.append(Label(self.classframe, text=n, font=("Arial Bold", 10)))
            self.labels[count].grid(row=0, column=count, padx=self.GRIDPADX, pady=self.GRIDPADY, sticky='nw')

    def generate(self):
        self.inf_img = next(iter(self.test_ds))

        photo = T.ToPILImage()(self.inf_img)
        photo = ImageTk.PhotoImage(photo)

        seg, over, cnames = self.inference()

        self.load_results(photo, seg, over, cnames)

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

        self.inf_img = self.transform(image)

        photo = ImageTk.PhotoImage(image)
        seg, over, cnames = self.inference()

        self.load_results(photo, seg, over, cnames)


app = App(appw=1400, appy=780)
app.mainloop()
