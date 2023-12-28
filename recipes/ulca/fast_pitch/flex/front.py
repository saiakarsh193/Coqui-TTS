import gradio as gr

MAX_PITCH_SLIDERS = 50

def after_variance_button():
    return [
        gr.update(visible=False), # variance button
        gr.update(visible=True), # speech button
        gr.update(visible=True) # clear button
    ]

def clear_button():
    updated_text = gr.update(value="")
    updated_hide = [gr.update(label=None, visible=False, interactive=False) for _ in range(MAX_PITCH_SLIDERS)]
    return [updated_text] + updated_hide

def after_clear_button():
    return [
        gr.update(visible=True), # variance button
        gr.update(visible=False), # speech button
        gr.update(visible=False) # clear button
    ]

def run_tts(text):
    tokens = list(text)
    tokens = tokens[: MAX_PITCH_SLIDERS]
    print(tokens)
    updated_show = [gr.update(label=tk, visible=True, interactive=True, value=10) for tk in tokens]
    updated_hide = [gr.update(label=None, visible=False, interactive=False) for _ in range(MAX_PITCH_SLIDERS - len(tokens))]
    return updated_show + updated_hide

with gr.Blocks(
    theme = "default",
    title = "FastPitchResearch"
    ) as demo:

    with gr.Row(
        variant="panel",
        ):
        gr.Markdown(
            f"""
            # Experimenting with FastPitch (Forward TTS)
            Sai Akarsh
            """
            )
    with gr.Row(
        variant = "panel"
        ):
        input_text = gr.Textbox(
            placeholder="we can generate speech by writing here!",
            label="input text",
            value="hello how are you"
            )
    # pitch sliders (max count of 50)
    psliders = []
    with gr.Column(
        variant = "panel"
        ):
        for _ in range(MAX_PITCH_SLIDERS):
            psliders.append(gr.Slider(
                label="",
                container=True,
                minimum=0,
                maximum=10,
                step=1,
                visible=False
                )
            )
    
    vbtn = gr.Button("Generate Pitch")
    sbtn = gr.Button("Generate Speech!", visible=False)
    cbtn = gr.Button("Clear", visible=False)

    vbtn.click(run_tts, input_text, psliders).then(after_variance_button , None, [vbtn, sbtn, cbtn])
    sbtn.click(lambda: , None, None).success(print("done succ")).
    cbtn.click(clear_button, None, [input_text] + psliders).then(after_clear_button, None, [vbtn, sbtn, cbtn])

demo.queue().launch(
    debug=True,
    # share = True
    )
