from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torch
import gdown
import os
from PIL import Image
import io

app = FastAPI()

MODEL_PATH = "model.pth"
FILE_ID = "1f66ewUKgn7CLJn0q74Mk_iBUGEdAz4sp"

@app.on_event("startup")
def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id=1f66ewUKgn7CLJn0q74Mk_iBUGEdAz4sp{FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()

@app.post("/process")
async def process(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    # TODO: Replace with your real preprocessing + model inference
    result = image

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)


    return StreamingResponse(buf, media_type="image/png")
