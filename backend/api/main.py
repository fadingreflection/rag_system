import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

from backend.model_inference.rag_inference import RagModelInference  # noqa: E402, I001

from fastapi import FastAPI  # noqa: E402

from pydantic import BaseModel  # noqa: E402
from backend.config import HOST, PORT  # noqa: E402

class ItemResponse(BaseModel):
    user_prompt: str
    model_response: str

app = FastAPI()
inst = RagModelInference()  # предполагается, что класс определён

@app.post("/rag_inference", response_model=ItemResponse)
def get_inference(user_prompt: str):
    resp = inst.rag_system_inference(user_prompt)["response"]
    return ItemResponse(user_prompt=user_prompt, model_response=resp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)