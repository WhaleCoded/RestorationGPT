from typing import Union, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CompletionRequest(BaseModel):
    text: Optional[str]
    context: Optional[str]

@app.get("/")
async def generate_response(request: Optional[CompletionRequest]):
    if request is None or not request.text:
        return {"response": "Please provide a text to complete, or a question you would like answered."}
    if request.context:
        text = request.context + request.text
    else:
        text = request.text