import typing as T

import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from lucidsonicdreams import LucidSonicDream
import io
import os
import openai


def render_text_to_clip() -> None:
    #st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":pencil2: Text to Audio & Video Clip")
    st.write(
        """
    Generate audio and visualization from text prompts.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool runs riffusion in the simplest text to image form to generate an audio
            clip from a text prompt. There is no seed image or interpolation here. This mode
            allows more diversity and creativity than when using a seed image, but it also
            leads to having less control. Play with the seed to get infinite variations.
            
            After the audio is generated, we run the LucidSonicDream to generate a visualization 
            for the given audio.
            """
        )

    device = streamlit_util.select_device(st.sidebar)
    extension = streamlit_util.select_audio_extension(st.sidebar)
    checkpoint = streamlit_util.select_checkpoint(st.sidebar)

    with st.form("Inputs"):
        openai_key = st.text_input("ChatGPT key")
        prompt = st.text_input("Prompt")
        negative_prompt = st.text_input("Negative prompt")

        row = st.columns(4)
        num_clips = T.cast(
            int,
            row[0].number_input(
                "Number of clips",
                value=1,
                min_value=1,
                max_value=25,
                help="How many outputs to generate (seed gets incremented)",
            ),
        )
        starting_seed = T.cast(
            int,
            row[1].number_input(
                "Seed",
                value=42,
                help="Change this to generate different variations",
            ),
        )

        st.form_submit_button("Riff", type="primary")

    with st.sidebar:
        num_inference_steps = T.cast(int, st.number_input("Inference steps", value=25))
        width = T.cast(int, st.number_input("Width", value=512))
        guidance = st.number_input(
            "Guidance", value=7.0, help="How much the model listens to the text prompt"
        )
        scheduler = st.selectbox(
            "Scheduler",
            options=streamlit_util.SCHEDULER_OPTIONS,
            index=0,
            help="Which diffusion scheduler to use",
        )
        assert scheduler is not None

        use_20k = st.checkbox("Use 20kHz", value=False)
        
        video_style = st.selectbox(
            "Style Model",
            options=os.listdir("../lsd-pytorch/weights/"),
            index=0,
            help="Which style model to use",
        )
        
        fpm = T.cast(int, st.number_input("Frames per Minute", value=8))
        pp = T.cast(bool, st.checkbox("Pulse Percusive", value=True))
        ph = T.cast(bool, st.checkbox("Pulse Harmonic", value=True))
        mr = T.cast(float, st.slider("Motion Randomness", min_value=0.1, max_value=1.0, value=0.5))
        vv = T.cast(float, st.slider("Variety", min_value=0.1, max_value=1.0, value=1.0))
        cc = T.cast(int, st.number_input("Class (-1 => All)", value=-1, min_value=-1, max_value=999))
        bs = T.cast(int, st.number_input("Batch Size", value=1, min_value=1, max_value=256))
        
        if(cc == -1):
            cc = None
        else:
            cc = [cc]
        
        assert scheduler is not None

    if not prompt:
        st.info("Enter a prompt")
        return

    if use_20k:
        params = SpectrogramParams(
            min_frequency=10,
            max_frequency=20000,
            sample_rate=44100,
            stereo=True,
        )
    else:
        params = SpectrogramParams(
            min_frequency=0,
            max_frequency=10000,
            stereo=False,
        )

    seed = starting_seed
    for i in range(1, num_clips + 1):
        st.write(f"#### Riff {i} / {num_clips} - Seed {seed}")

        image = streamlit_util.run_txt2img(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=512,
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )
        st.image(image)
        
        

        segment = streamlit_util.audio_segment_from_spectrogram_image(
            image=image,
            params=params,
            device=device,
        )
        
        audio_bytes = io.BytesIO()
        segment.export(audio_bytes, format="mp3")
        
        
        #segment.export("song.mp3", format="mp3")
        with streamlit_util.pipeline_lock():
            L = LucidSonicDream(song = audio_bytes, 
                                style = '../lsd-pytorch/weights/' + video_style)
                                
            L.hallucinate(file_name = 'song.mp4',
                        speed_fpm = fpm,
                        pulse_percussive = pp,
                        pulse_harmonic = ph,
                        motion_randomness = mr,
                        truncation = vv,
                        classes = cc,
                        batch_size = bs)
            
            video_file = open('song.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            del L
       
        
        #streamlit_util.display_and_download_audio(
        #    segment, name=f"{prompt.replace(' ', '_')}_{seed}", extension=extension
        #)
        if openai_key:
            openai.api_key = openai_key
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Create a lyric poem for " + prompt}])
            st.text(completion.choices[0].message.content)

        seed += 1


if __name__ == "__main__":
    render_text_to_clip()
