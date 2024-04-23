# Version: 3.41.2

import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image
import rembg
import tempfile
import argparse
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import numpy as np
from diffusion_webui.diffusion_models.base_controlnet_pipeline import (
    ControlnetPipeline,
)
from diffusion_webui.utils.model_list import (
    controlnet_model_list,
    stable_model_list,
)
from diffusion_webui.utils.preprocces_utils import PREPROCCES_DICT
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_MAPPING,
    get_scheduler,
)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()

def check_input_image(self, input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image.astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(input_image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

def generate(image, mc_resolution, formats=["obj"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb = generate(preprocessed, 256, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb

class StableDiffusionControlNetGenerator(ControlnetPipeline):
    def __init__(self):
        self.pipe = None

    def load_model(self, stable_model_path, controlnet_model_path, scheduler):
        if self.pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path, torch_dtype=torch.float16, cache_dir="./model/"
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=stable_model_path,
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16,
                cache_dir="./model/"
            )

        self.pipe = get_scheduler(pipe=self.pipe, scheduler=scheduler)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def controlnet_preprocces(
        self,
        read_image: str,
        preprocces_type: str,
    ):
        processed_image = PREPROCCES_DICT[preprocces_type](read_image)
        return processed_image


    def generate_image(
        self,
        image_path: str,
        stable_model_path: str,
        controlnet_model_path: str,
        height: int,
        width: int,
        guess_mode: bool,
        controlnet_conditioning_scale: int,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        guidance_scale: int,
        num_inference_step: int,
        scheduler: str,
        seed_generator: int,
        preprocces_type: str = "Lineart",
    ):
        
        pipe = self.load_model(
            stable_model_path=stable_model_path,
            controlnet_model_path=controlnet_model_path,
            scheduler=scheduler,
        )

        read_image = Image.open(image_path)
        controlnet_image = self.controlnet_preprocces(
            read_image=read_image, preprocces_type=preprocces_type
        )

        if seed_generator == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(seed_generator)

        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            guess_mode=guess_mode,
            image=controlnet_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images
        return output[0]

    def app():
            with gr.Blocks():
                with gr.Row():
                    with gr.Column():
                        controlnet_image_path = gr.Image(
                            type="filepath", label="Image", height=400
                        )
                    
                    with gr.Column():
                        controlnet_prompt = gr.Textbox(
                            lines=1, placeholder="Prompt", show_label=False
                        )
                        controlnet_negative_prompt = gr.Textbox(
                            lines=1, placeholder="Negative Prompt", show_label=False
                        )

                        with gr.Row():
                            with gr.Column():
                                controlnet_stable_model_path = gr.Dropdown(
                                    choices=stable_model_list,
                                    value=stable_model_list[0],
                                    label="Stable Model Path",
                                )
                                controlnet_preprocces_type = gr.Dropdown(
                                    choices=list(PREPROCCES_DICT.keys()),
                                    value="Lineart",
                                    label="Preprocess Type",
                                    visible = False
                            )
                                controlnet_conditioning_scale = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.1,
                                    value=1.0,
                                    label="ControlNet Conditioning Scale",
                                )
                                controlnet_guidance_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=15,
                                    step=0.1,
                                    value=7.5,
                                    label="Guidance Scale",
                                )
                            
                                controlnet_width = gr.Slider(
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                    label="Width",
                                )

                            with gr.Row():
                                with gr.Column():
                                    controlnet_model_path = gr.Dropdown(
                                        choices=controlnet_model_list,
                                        value="lllyasviel/control_v11p_sd15_lineart",
                                        label="ControlNet Model Path",
                                        visible=False
                                    )
                                    controlnet_scheduler = gr.Dropdown(
                                        choices=list(SCHEDULER_MAPPING.keys()),
                                        value=list(SCHEDULER_MAPPING.keys())[0],
                                        label="Scheduler",
                                    )
                                    controlnet_num_inference_step = gr.Slider(
                                        minimum=1,
                                        maximum=150,
                                        step=1,
                                        value=30,
                                        label="Num Inference Step",
                                    )
                                    controlnet_num_images_per_prompt = gr.Slider(
                                        minimum=1,
                                        maximum=4,
                                        step=1,
                                        value=1,
                                        label="Number Of Images",
                                        visible=False
                                    )   
                                    controlnet_seed_generator = gr.Slider(
                                        minimum=0,
                                        maximum=1000000,
                                        step=1,
                                        value=0,
                                        label="Seed(0 for random)",
                                    )
                                    controlnet_guess_mode = gr.Checkbox(
                                        label="Guess Mode",
                                        value=True,
                                        visible=False
                                    )
                                    controlnet_height = gr.Slider(
                                        minimum=128,
                                        maximum=1280,
                                        step=32,
                                        value=512,
                                        label="Height",
                                    )
                                    

                with gr.Row():
                    with gr.Column():
                        # Gallery to display the generated images
                        output_image = gr.Image(
                            label="Generated images",
                            show_label=False,
                            # interactive=False,
                            height=350,
                            image_mode="RGBA",
                            type="pil"
                        )
                        # Button to generate the image
                        predict_button = gr.Button(value="Generate Image")
                    
                    with gr.Column():
                        # To display the 3D model
                        output_model_obj = gr.Model3D(
                                label="Output Model (OBJ Format)",
                                interactive=False,
                                height=500
                        )
                        # Button to generate the image
                        predict_button_triposr = gr.Button(value="Generated SD")
                with gr.Row():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=320,
                        value=256,
                        step=32
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
    
            predict_button.click(
                fn=StableDiffusionControlNetGenerator().generate_image,
                inputs=[
                    controlnet_image_path,
                    controlnet_stable_model_path,
                    controlnet_model_path,
                    controlnet_height,
                    controlnet_width,
                    controlnet_guess_mode,
                    controlnet_conditioning_scale,
                    controlnet_prompt,
                    controlnet_negative_prompt,
                    controlnet_num_images_per_prompt,
                    controlnet_guidance_scale,
                    controlnet_num_inference_step,
                    controlnet_scheduler,
                    controlnet_seed_generator,
                    controlnet_preprocces_type,
                ],
                outputs=[output_image],
            )

            predict_button_triposr.click(
                fn=preprocess,
                inputs=[output_image, do_remove_background, foreground_ratio],
                outputs=[processed_image],
            )