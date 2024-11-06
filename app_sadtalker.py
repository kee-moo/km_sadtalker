import os
import shutil
import sys
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import gradio as gr
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional, Dict
import logging

from fastapi.responses import FileResponse

from src.gradio_demo import SadTalker

app = FastAPI()

log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置日志文件路径，并按日期和小时命名
current_time = datetime.now().strftime("%Y-%m-%d-%H")
log_file = os.path.join(log_dir, f'info-{current_time}.log')

# 设置日志文件按小时切分
log_handler = TimedRotatingFileHandler(
    log_file, when='H', interval=1, backupCount=24
)

log_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

# 配置日志
logging.basicConfig(level=logging.DEBUG, handlers=[log_handler])

task_results: Dict[str, Optional[str]] = {}
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_PATH, "results/")


class PrintLogger:
    def write(self, message):
        if message.strip():  # 去掉多余的换行
            logging.info(message.strip())  # 记录为 INFO 级别日志

    def flush(self):
        pass  # 让 sys.stdout 能正常刷新

    def isatty(self):
        return True  # 返回 True，使其兼容 uvicorn 的 TTY 检查


# 将 sys.stdout 重定向为 PrintLogger
sys.stdout = PrintLogger()

try:
    import webui  # in webui

    in_webui = True
except:
    in_webui = False


def toggle_audio_file(choice):
    if not choice:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)


def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")

        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", type="filepath", elem_id="img2img_image")

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", type="filepath")

                        if sys.platform != 'win32' and not in_webui:
                            from src.utils.text2speech import TTSTalker
                            tts_talker = TTSTalker()
                            with gr.Column(variant='panel'):
                                input_text = gr.Textbox(label="Generating audio from text", lines=5,
                                                        placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                                tts = gr.Button('Generate audio', elem_id="sadtalker_audio_generate", variant='primary')
                                tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        gr.Markdown(
                            "need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more detials")
                        with gr.Column(variant='panel'):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0)  #
                            size_of_image = gr.Radio([256, 512], value=256, label='face model resolution',
                                                     info="use 256/512 model?")  #
                            preprocess_type = gr.Radio(['crop', 'resize', 'full', 'extcrop', 'extfull'], value='crop',
                                                       label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(
                                label="Still Mode (fewer head motion, works with preprocess `full`)")
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2)
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(label="Generated video", format="mp4")

        if warpfn:
            submit.click(
                fn=warpfn(sad_talker.test),
                inputs=[source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,
                        size_of_image,
                        pose_style
                        ],
                outputs=[gen_video]
            )
        else:
            submit.click(
                fn=sad_talker.test,
                inputs=[source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,
                        size_of_image,
                        pose_style
                        ],
                outputs=[gen_video]
            )

    return sadtalker_interface


class ResponseModel(BaseModel):
    code: int
    message: str
    body: Any


def create_response(code: int, message: str, body: Any = None) -> JSONResponse:
    return JSONResponse(content=ResponseModel(code=code, message=message, body=body).model_dump())


def gradio_interface(source_image, driven_audio, *args):
    task_id = generate_task(source_image, driven_audio, *args)
    return f"Task ID: {task_id}"


def generate_task(task_id, source_image_path, driven_audio_path, **kwargs):
    sad_talker = SadTalker()
    result_path = sad_talker.test(source_image_path, driven_audio_path, **kwargs)
    logging.info("task is in process, result path: {}".format(result_path))
    task_results[task_id] = result_path
    return task_id


class Params(BaseModel):
    preprocess: str = 'crop'
    still_mode: bool = False
    use_enhancer: bool = False
    batch_size: int = 1
    size: int = 256
    pose_style: int = 0
    exp_scale: float = 1.0
    use_ref_video: bool = False
    ref_video: Optional[str] = None
    ref_info: Optional[str] = None
    use_idle_mode: bool = False
    length_of_audio: int = 0
    use_blink: bool = True
    result_dir: str = './results/'


@app.post("/km_sadtalker/generate")
async def generate(
        background_tasks: BackgroundTasks,
        source_image: UploadFile = File(...),
        driven_audio: UploadFile = File(...),
        preprocess: str = Form('crop'),
        still_mode: bool = Form(False),
        use_enhancer: bool = Form(False),
        batch_size: int = Form(1),
        size: int = Form(256),
        pose_style: int = Form(0),
        exp_scale: float = Form(1.0),
        use_ref_video: bool = Form(False),
        ref_video: str = Form(None),
        ref_info: str = Form(None),
        use_idle_mode: bool = Form(False),
        length_of_audio: int = Form(0),
        use_blink: bool = Form(True),
        result_dir: str = Form('./results/')
):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    source_image_path = f"./temp/{source_image.filename}"
    with open(source_image_path, "wb") as buffer:
        shutil.copyfileobj(source_image.file, buffer)

    driven_audio_path = f"./temp/{driven_audio.filename}"
    with open(driven_audio_path, "wb") as buffer:
        shutil.copyfileobj(driven_audio.file, buffer)
    task_id = str(uuid.uuid4())
    background_tasks.add_task(generate_task, task_id, source_image_path, driven_audio_path,
                              preprocess=preprocess,  # 参数4及以后
                              still_mode=still_mode,
                              use_enhancer=use_enhancer,
                              batch_size=batch_size,
                              size=size,
                              pose_style=pose_style,
                              exp_scale=exp_scale,
                              use_ref_video=use_ref_video,
                              ref_video=ref_video,
                              ref_info=ref_info,
                              use_idle_mode=use_idle_mode,
                              length_of_audio=length_of_audio,
                              use_blink=use_blink,
                              result_dir=result_dir)
    return create_response(0, "ok", {"task_id": task_id})


@app.get("/km_sadtalker/task/status")
async def get_task_status(task_id: str):
    result = task_results.get(task_id)
    if result is None:
        return create_response(1, "fail", "task not found")
    elif isinstance(result, str):
        return create_response(0, "ok", "task is complete")
    else:
        return create_response(2, "fail", "task not exist")


@app.get("/km_sadtalker/download")
async def download(task_id: str):
    result_path = task_results.get(task_id)
    logging.info("download result path: {}".format(result_path))
    if result_path is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Task not found")
    return FileResponse(result_path, media_type="application/octet-stream", filename=os.path.basename(result_path))


if __name__ == "__main__":
    demo = sadtalker_demo()
    demo.queue()
    app = gr.mount_gradio_app(app, demo, path="/home")
    uvicorn.run(app, port=6006)
