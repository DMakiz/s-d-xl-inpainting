import gradio as gr
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting, UNet2DConditionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def predict(dict):
    invert_mask = dict["invert_mask"]
    image = dict["image"]
    init_image = Image.open(image).convert("RGB").resize((1024, 1024))
    
    if dict["output_type"] == "image":
        output = init_image
    else:
        mask = generate_mask(init_image, invert_mask)
        output = mask
    
    output = pipe(prompt=dict["prompt"], negative_prompt=dict["negative_prompt"], image=init_image, mask_image=output, guidance_scale=dict["guidance_scale"], num_inference_steps=int(dict["steps"]), strength=dict["strength"])
    
    return output.images[0]

css = '''
..gradio-container{max-width: 1100px !important}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; max-width: 13rem; margin-left: auto;}
div#share-btn-container > div {flex-direction: row;background: black;align-items: center}
#share-btn-container:hover {background-color: #060606}
#share-btn {all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;right:0;}
#share-btn * {all: unset}
#share-btn-container div:nth-child(-n+2){width: auto !important;min-height: 0px !important;}
#share-btn-container .wrap {display: none !important}
#share-btn-container.hidden {display: none!important}
#prompt input{width: calc(100% - 160px);border-top-right-radius: 0px;border-bottom-right-radius: 0px;}
#run_button{position:absolute;margin-top: 11px;right: 0;margin-right: 0.8em;border-bottom-left-radius: 0px;
    border-top-left-radius: 0px;}
#prompt-container{margin-top:-18px;}
#prompt-container .form{border-top-left-radius: 0;border-top-right-radius: 0}
#image_upload{border-bottom-left-radius: 0px;border-bottom-right-radius: 0px}
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
                output_type_radio = gr.Radio(["Image", "Mask"], label="Output Type", initial_value="Image",
                                            elem_id="output_type_radio")
            
            with gr.Row():
                with gr.Column():
                    guidance_scale_slider = gr.Slider(label="Guidance Scale", min_value=0, max_value=1, step_size=0.1, 
                                                     initial_value=0.5, elem_id='guidance_scale_slider')
                    num_steps_slider = gr.Slider(label="Number of Inference Steps", min_value=1, max_value=10, step_size=1, 
                                                 initial_value=5, elem_id='num_steps_slider')
                with gr.Column():
                    strength_slider = gr.Slider(label="Strength", min_value=0.0, max_value=1.0, step_size=0.1, 
                                                initial_value=0.3, elem_id='strength_slider')
        
        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", height=400)
    
    btn.click(fn=predict, inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider], 
              outputs=[image_out])
    invert_mask_checkbox.change(fn=predict, inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider],
                                outputs=[image_out])
    prompt.submit(fn=predict, inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider],
                  outputs=[image_out])
    output_type_radio.click(fn=predict, inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider],
                            outputs=[image_out])
    guidance_scale_slider.change(fn=predict, inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider],
                                 outputs=[image_out])
    num_steps_slider.change(fn=predict, inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider],
                            outputs=[image_out])
    strength_slider.change(fn=predict, inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider],
                           outputs=[image_out])
    
    gr.Examples(
        examples=[
            {"image": "./imgs/aaa (8).png"},
            {"image": "./imgs/download (1).jpeg"},
            {"image": "./imgs/0_oE0mLhfhtS_3Nfm2.png"},
            {"image": "./imgs/02_HubertyBlog-1-1024x1024.jpg"},
            {"image": "./imgs/jdn_jacques_de_nuce-1024x1024.jpg"},
            {"image": "./imgs/c4ca473acde04280d44128ad8ee09e8a.jpg"},
            {"image": "./imgs/canam-electric-motorcycles-scaled.jpg"},
            {"image": "./imgs/e8717ce80b394d1b9a610d04a1decd3a.jpeg"},
            {"image": "./imgs/Nature___Mountains_Big_Mountain_018453_31.jpg"},
            {"image": "./imgs/Multible-sharing-room_ccexpress-2-1024x1024.jpeg"},
        ],
        fn=predict,
        inputs=[image, invert_mask_checkbox, prompt, output_type_radio, guidance_scale_slider, num_steps_slider, strength_slider],
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
