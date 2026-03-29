from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "API Service running"}

@app.post("/ask")
def ask(req: QueryRequest):

    try:
        print("Calling RAG service...")
        response = requests.post(
            "http://rag:8001/rag",
            json={"question": req.question}
        )

        print("RAG RESPONSE:", response.text) 

        rag_result = response.json()
        if "error" in rag_result:
            return {"error": f"RAG service error: {rag_result['error']}"}
        return {
            "question": req.question,
            "answer": rag_result["answer"]   # ✅ safe now
        }

    except Exception as e:
        return {"error": str(e)}