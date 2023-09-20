import gradio as gr
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
import diffusers
from share_btn import community_icon_html, loading_icon_html, share_js
from PIL import Image, ImageOps

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def predict(dict, invert_mask=False, prompt="", negative_prompt="", guidance_scale=7.5, steps=20, strength=1.0, scheduler="EulerDiscreteScheduler"):
    if negative_prompt == "":
        negative_prompt = None
    scheduler_class_name = scheduler.split("-")[0]

    add_kwargs = {}
    if len(scheduler.split("-")) > 1:
        add_kwargs["use_karras"] = True
    if len(scheduler.split("-")) > 2:
        add_kwargs["algorithm_type"] = "sde-dpmsolver++"

    scheduler = getattr(diffusers, scheduler_class_name)
    pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **add_kwargs)

    init_image = dict["image"].convert("RGB").resize((1024, 1024))
    mask = dict["mask"].convert("RGB").resize((1024, 1024))
    
    if invert_mask:
        mask = ImageOps.invert(mask)
    
    output = mask
    return output, gr.update(visible=True)


css = '''

'''

image_blocks = gr.Blocks(css=css, elem_id="total-container")
with image_blocks as demo:
    gr.HTML(read_content("header.html"))
    with gr.Row():
        with gr.Column():
            image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload", height=400)
            with gr.Row(elem_id="prompt-container", mobile_collapse=False, equal_height=True):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Your prompt (what you want in place of what is erased)", show_label=False, elem_id="prompt")
                    btn = gr.Button("Inpaint!", elem_id="run_button")
            
            with gr.Row(mobile_collapse=False, equal_height=True):
                invert_mask_checkbox = gr.Checkbox(label="Invert Mask", initial_value=False, elem_id="invert_mask_checkbox")
                guidance_scale = gr.Number(value=7.5, minimum=1.0, maximum=20.0, step=0.1, label="guidance_scale")
                steps = gr.Number(value=20, minimum=10, maximum=30, step=1, label="steps")
                strength = gr.Number(value=0.99, minimum=0.01, maximum=0.99, step=0.01, label="strength")
                negative_prompt = gr.Textbox(label="negative_prompt", placeholder="Your negative prompt", info="what you don't want to see in the image")
            with gr.Row(mobile_collapse=False, equal_height=True):
                schedulers = ["DEISMultistepScheduler", "HeunDiscreteScheduler", "EulerDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler-Karras", "DPMSolverMultistepScheduler-Karras-SDE"]
                scheduler = gr.Dropdown(label="Schedulers", choices=schedulers, value="EulerDiscreteScheduler")
        
        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            with gr.Group(elem_id="share-btn-container", visible=False) as share_btn_container:
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
            

    btn.click(fn=predict, inputs=[image, invert_mask_checkbox, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=[image_out, share_btn_container], api_name='run')
    invert_mask_checkbox.change(fn=predict, inputs=[image, invert_mask_checkbox, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=[image_out, share_btn_container])
    prompt.submit(fn=predict, inputs=[image, invert_mask_checkbox, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=[image_out, share_btn_container])
    share_button.click(None, [], [], _js=share_js)

    gr.Examples(
        examples=[
            ["./imgs/aaa (8).png"],
            ["./imgs/download (1).jpeg"],
            ["./imgs/0_oE0mLhfhtS_3Nfm2.png"],
            ["./imgs/02_HubertyBlog-1-1024x1024.jpg"],
            ["./imgs/jdn_jacques_de_nuce-1024x1024.jpg"],
            ["./imgs/c4ca473acde04280d44128ad8ee09e8a.jpg"],
            ["./imgs/canam-electric-motorcycles-scaled.jpg"],
            ["./imgs/e8717ce80b394d1b9a610d04a1decd3a.jpeg"],
            ["./imgs/Nature___Mountains_Big_Mountain_018453_31.jpg"],
            ["./imgs/Multible-sharing-room_ccexpress-2-1024x1024.jpeg"],
        ],
        fn=predict,
        inputs=[image, invert_mask_checkbox],
        cache_examples=False,
    )
    gr.HTML(
        """
        <div class="footer">
            <p>Model by <a href="https://huggingface.co/diffusers" style="text-decoration: underline;" target="_blank">Diffusers</a> - Gradio Demo by 🤗 Hugging Face
            </p>
        </div>
        """
    )

image_blocks.queue(max_size=25).launch(debug=True, max_threads=True, share=True, inbrowser=True)
