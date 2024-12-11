import panel as pn
import os
from PIL import Image
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, BoxEditTool, CheckboxGroup, CustomJS, LinearColorMapper
from scipy.ndimage import sobel, gaussian_filter

# Enable Panel extension
pn.extension()

# Base directory for case data
BASE_DIR = r"C:\Users\syed_fahim_ahmed\Documents\data\Cases"

# Function to retrieve available cases
def get_cases():
    try:
        return [case for case in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, case))]
    except Exception as e:
        return [f"Error: {str(e)}"]

# Function to retrieve panels for a given case
def get_panels(case):
    try:
        case_dir = os.path.join(BASE_DIR, case)
        files = os.listdir(case_dir)
        panels = set("_".join(f.split("_")[:3]) for f in files if f.endswith(".tif"))
        return sorted(panels)
    except Exception as e:
        return [f"Error: {str(e)}"]
    
# Function to compute gradient magnitude
def gradmag(n):
    sobel_x = sobel(n, axis=0)  # Gradient along x-axis
    sobel_y = sobel(n, axis=1)  # Gradient along y-axis
    return np.sqrt(sobel_x**2 + sobel_y**2)

# Function to process and display grayscale and RGB images
def display_images_with_checkboxes(case, panel):
    if not panel:
        return pn.pane.Markdown("### No panel selected.", style={'color': 'red'})

    case_dir = os.path.join(BASE_DIR, case)
    files = [f for f in os.listdir(case_dir) if f.startswith(panel) and f.endswith(".tif")]
    
    if not files:
        return pn.pane.Markdown("### No images available for the selected panel.", style={'color': 'red'})

    # Load images for wavelengths 470 and 528
    images = {}
    for file in files:
        wavelength = file.split("_")[-1].replace(".tif", "")
        img_path = os.path.join(case_dir, file)
        img = Image.open(img_path)
        images[wavelength] = img

    if "470" in images and "528" in images:
        # Process images for display
        img_470 = np.array(images["470"], dtype=np.float32)
        img_528 = np.array(images["528"], dtype=np.float32)
        
        low_in_470 = np.min(img_470)
        high_in_470 = np.max(img_470)

        low_in_528 = np.min(img_528)
        high_in_528 = np.max(img_528)

        cI = 0.8
        nI = 0.8
        cG = 0.7
        nG = 0.8

        def gammaScale(image, gamma, low_in, high_in):
            new_img = np.power(np.divide(np.subtract(image, low_in), high_in - low_in), gamma)
            return new_img

        def imageScale(image, scale):
            new_img = np.multiply(image, scale)
            return new_img

        tile_red = imageScale(img_470, nI)
        tile_red = gammaScale(img_470, nG, low_in_470, high_in_470)

        tile_blue = imageScale(img_528, cI)
        tile_blue = gammaScale(img_528, cG, low_in_528, high_in_528)



        def ibMGColorReMap(ired, iblue):
            B_Beta_E = 0.7000
            B_Beta_H = 0.5000
            c = 0.0821
            G_Beta_E = 0.9000
            G_Beta_H = 1.0
            k = 1.0894
            R_Beta_E = 0.0200
            R_Beta_H = 0.8600

            I_r = ((np.exp(-R_Beta_H * ired * 2.5) - c) * k) * ((np.exp(-R_Beta_E * iblue * 2.5) - c) * k)
            I_g = ((np.exp(-G_Beta_H * ired * 2.5) - c) * k) * ((np.exp(-G_Beta_E * iblue * 2.5) - c) * k)
            I_b = ((np.exp(-B_Beta_H * ired * 2.5) - c) * k) * ((np.exp(-B_Beta_E * iblue * 2.5) - c) * k)

            rgb = np.squeeze(np.stack((I_r, I_g, I_b), axis=-1))
            rgb *= 255
            rgb = rgb.astype("uint8")
            return rgb

        #rgb = ibMGColorReMap(tile_red, tile_blue)

        rgb_image = ibMGColorReMap(tile_red, tile_blue)
        gray_image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
        
        # Compute gradient magnitude
        gradient_magnitude = gradmag(gaussian_filter(gray_image, sigma=5))
        gradient_magnitude = 255.0 - (255.0 * (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.ptp())).astype(np.float32)
        
        def to_rgba(rgb):
            """Convert RGB image to RGBA format compatible with Bokeh."""
            X, Y, _ = rgb.shape
            # Step 1: Add an alpha channel
            rgba_image = np.dstack((rgb, 255 * np.ones((X, Y), dtype=np.uint8)))
            # Step 2: Flatten RGBA to uint32 format
            rgba_flat = np.zeros((X, Y), dtype=np.uint32)
            view = rgba_flat.view(dtype=np.uint8).reshape((X, Y, 4))
            view[:, :, :] = rgba_image
            return rgba_flat

        rgba_image = to_rgba(rgb_image)

        # Full RGBA image figure
        rgba_fig = figure(
            title=f"{panel} - RGBA Image",
            x_range=(0, rgba_image.shape[1]),
            y_range=(0, rgba_image.shape[0]),
            tools="pan,box_zoom,reset",
            toolbar_location="below",
            width=512,
            height=512
        )
        rgba_source = ColumnDataSource(data=dict(image=[rgba_image]))
        rgba_fig.image_rgba(
            image='image',
            x=0, y=0,
            dw=rgba_image.shape[1],
            dh=rgba_image.shape[0],
            source=rgba_source
        )

        # Add bounding box for cropping
        box_source = ColumnDataSource(data=dict(x=[256], y=[256], width=[512], height=[512]))
        renderer = rgba_fig.rect(
            x='x', y='y', width='width', height='height', source=box_source,
            line_color="red", fill_alpha=0.2
        )
        box_edit = BoxEditTool(renderers=[renderer], num_objects=1)
        rgba_fig.add_tools(box_edit)
        rgba_fig.toolbar.active_drag = box_edit

        # Cropped image figure
        cropped_source = ColumnDataSource(data=dict(image=[], gray=[], x=[0], y=[0], dw=[512], dh=[512]))
        cropped_fig = figure(
            title=f"{panel} - Cropped Image",
            x_range=(0, 512),
            y_range=(0, 512),
            tools="pan,box_zoom,reset",
            toolbar_location="below",
            width=512,
            height=512
        )
        rgba_renderer = cropped_fig.image_rgba(
            image='image', x='x', y='y', dw='dw', dh='dh', source=cropped_source, visible=True
        )
        gray_mapper = LinearColorMapper(palette="Greys256", low=0, high=255)
        gray_renderer = cropped_fig.image(
            image='gray', x='x', y='y', dw='dw', dh='dh', source=cropped_source,
            color_mapper=gray_mapper, visible=False
        )
        grad_mapper = LinearColorMapper(palette="Greys256", low=np.min(gradient_magnitude), high=np.max(gradient_magnitude))
        grad_renderer = cropped_fig.image(image='grad', x='x', y='y', dw='dw', dh='dh', source=cropped_source, color_mapper=grad_mapper, visible=False)

        def update_cropped(attr, old, new):
            if box_source.data['x']:
                x_center = box_source.data['x'][0]
                y_center = box_source.data['y'][0]
                width = box_source.data['width'][0]
                height = box_source.data['height'][0]

                x0 = max(0, x_center - width / 2)
                y0 = max(0, y_center - height / 2)
                x1 = min(rgba_image.shape[1], x_center + width / 2)
                y1 = min(rgba_image.shape[0], y_center + height / 2)

                cropped_rgba = rgba_image[int(y0):int(y1), int(x0):int(x1)]
                cropped_gray = gray_image[int(y0):int(y1), int(x0):int(x1)]
                cropped_grad = gradient_magnitude[int(y0):int(y1), int(x0):int(x1)]

                cropped_source.data = {
                    'image': [cropped_rgba],
                    'gray': [cropped_gray],
                    'grad': [cropped_grad],
                    'x': [0],
                    'y': [0],
                    'dw': [cropped_rgba.shape[1]],
                    'dh': [cropped_rgba.shape[0]]
                }

        box_source.on_change('data', update_cropped)

        # Checkbox for toggling visibility
        checkbox = CheckboxGroup(labels=["Show RGBA Image", "Show Grayscale Image", "Show Transformed Scalar Image"], active=[0])
        checkbox.js_on_change(
            'active',
            CustomJS(args=dict(rgba_renderer=rgba_renderer, gray_renderer=gray_renderer, grad_renderer=grad_renderer), code="""
                rgba_renderer.visible = cb_obj.active.includes(0);
                gray_renderer.visible = cb_obj.active.includes(1);
                grad_renderer.visible = cb_obj.active.includes(2);
            """)
        )

        # Return layout
        return pn.Row(
            rgba_fig,
            pn.Column(cropped_fig, checkbox)
        )

case_dropdown = pn.widgets.Select(name="Select Case", options=get_cases())
panel_dropdown = pn.widgets.Select(name="Select Panel", options=[])

def update_panel_dropdown(event):
    panels = get_panels(case_dropdown.value)
    panel_dropdown.options = panels
    panel_dropdown.value = panels[0] if panels else None

case_dropdown.param.watch(update_panel_dropdown, "value")
update_panel_dropdown(None)

image_viewer = pn.bind(display_images_with_checkboxes, case=case_dropdown, panel=panel_dropdown)

app = pn.Column(
    pn.Row(case_dropdown, panel_dropdown),
    image_viewer
)

app.servable()