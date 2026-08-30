from fastapi import FastAPI

app = FastAPI()

@app.get("/multiple/{number}")
async def multiple(number: float):
    return {"multiple": number ** number}
