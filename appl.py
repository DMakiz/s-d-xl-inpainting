import gradio as gr
from PIL import Image, ImageOps

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def generate_mask(image):
    return image


css = '''
.gradio-container{max-width: 1100px !important}
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
            with gr.Row():
                invert_mask_checkbox = gr.Checkbox(label="Invert Mask", initial_value=False, elem_id="invert_mask_checkbox")
                generate_button = gr.Button("Generate Mask", elem_id="generate_button")
        with gr.Column():
            mask_out = gr.Image(label="Mask Output", elem_id="mask_output", height=1024)
    
    generate_button.click(generate_mask, inputs=[image], outputs=[mask_out])

    gr.HTML(
        """
        <div class="footer">
            <p>Model by <a href="https://huggingface.co/diffusers" style="text-decoration: underline;" target="_blank">Diffusers</a> - Gradio Demo by 🤗 Hugging Face
            </p>
        </div>
        """
    )

image_blocks.queue(max_size=25).launch(debug=True, max_threads=True, share=True, inbrowser=True)
