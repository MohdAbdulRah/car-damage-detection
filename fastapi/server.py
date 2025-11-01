from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model_helper import predict

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def get_prediction(file:UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_path = "temp_file.jpg"
        with open(image_path,"wb") as f:
            f.write(image_bytes)
        prediction = predict(image_path)
        return {"Prediction" : prediction}
    except Exception as e:
        return {"Error" : str(e)}