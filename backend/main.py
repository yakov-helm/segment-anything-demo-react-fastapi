from fastapi import FastAPI, Request, Response
import uvicorn
from starlette.middleware.cors import CORSMiddleware

import os
import functools

from embedded import make_embedding

app = FastAPI()

GALLERY = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/frontend/src/assets/gallery"


@functools.lru_cache()
@app.get("/ai/list")
async def list_files(request: Request):
    body = await request.body()
    file_names = [fname for fname in os.listdir(GALLERY) if fname.endswith('.jpg') or fname.endswith('.png')]
    return {"files": sorted(file_names)}


@functools.lru_cache()
@app.post("/ai/embedded/{file_name}")
async def embedded(request: Request, file_name:str):
    body = await request.body()
    
    root = f"{GALLERY}/{file_name}.npy"
    checkpoint = "sam_vit_b.pth"
    model_type = "vit_b"
    print(root)
    make_embedding(body, root, checkpoint, model_type)
    
    return {"npy": f"{file_name}.npy"}


@app.post("/ai/embedded/all/{file_name}")
async def embedded(request: Request, file_name:str):
    return Response(status_code=200)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host="localhost",
        port=8000,
        reload=True,
    )
