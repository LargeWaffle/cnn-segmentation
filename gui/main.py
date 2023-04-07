import dearpygui.dearpygui as dpg
from PIL import Image
from pytorch import run_segmentation
import numpy as np

W_WIDTH = 1280
W_HEIGHT = 800

IMG_W = int(W_WIDTH * 0.8)
SIDE_W = int(W_WIDTH * 0.2)

POP_W = int(W_WIDTH // 1.3)
POP_H = int(W_HEIGHT // 1.3)

dpg.create_context()


def add_and_load_image(image, tag, parent=None):
    w, h = image.size

    image.putalpha(255)
    dpg_image = np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0

    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_static_texture(w, h, dpg_image, parent=reg_id)

    if parent:
        return dpg.add_image(texture_id, tag=tag, parent=parent)
    else:
        return dpg.add_image(texture_id, tag=tag)


def load_imgs(_, app_data):
    with dpg.value_registry():
        dpg.add_string_value(default_value=app_data['file_path_name'], tag="base_path")

    img = Image.open(dpg.get_value("base_path"))
    res_img = run_segmentation(img)

    add_and_load_image(image=img, tag="base_img", parent=image_window)
    add_and_load_image(image=res_img, tag="res_img", parent=image_window)


def rerun():
    dpg.delete_item("res_img")

    img = Image.open(dpg.get_value("base_path"))
    res_img = run_segmentation(img)

    add_and_load_image(image=res_img, tag="res_img", parent=image_window)


def cancel_callback():
    pass


with dpg.file_dialog(directory_selector=False, show=False, id="filepicker", callback=load_imgs,
                     cancel_callback=cancel_callback,
                     width=POP_W, height=POP_H):
    dpg.add_file_extension("Images (*.png *.jpg *.jpeg){.png,.jpg,.jpeg}", color=(255, 255, 255, 255))

with dpg.window(width=W_WIDTH, height=W_HEIGHT, autosize=True) as main_window:

    with dpg.child_window(pos=[0, 0], width=IMG_W, autosize_x=True, autosize_y=True, parent=main_window)as image_window:
        dpg.add_text("Segmentation results", tag="img_res_title")
        dpg.add_spacer(height=15)

    with dpg.child_window(pos=[IMG_W, 0], width=SIDE_W, autosize_x=True, autosize_y=True, parent=main_window):
        dpg.add_button(label="Choose Image", tag="file_button", callback=lambda: dpg.show_item("filepicker"), width=-1)
        dpg.add_spacer(height=5)
        dpg.add_button(label="Regenerate", tag="seg_button", callback=rerun, width=-1)

dpg.create_viewport(title='Image segmentation', width=W_WIDTH, height=W_HEIGHT, x_pos=0, y_pos=0)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window(window=main_window, value=True)
dpg.start_dearpygui()

dpg.destroy_context()
