from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM
import torch
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

class TaggingRequest(BaseModel):
    texts: list[str]

class TaggingResponse(BaseModel):
    tags: list[str]

class TagPredictor:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        self.generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.1,
            max_new_tokens=int(os.getenv('MAX_NEW_TOKENS')),
            pad_token_id=self.tokenizer.eos_token_id
        )

    def process_input(self, text):
        return f"""<s>[INST] You are a careful data annotator. Your task is to assign up to five unique and relevant tags to the provided text without explanations. Return the tags in the format: ['Tag1','Tag2',...,'Tag5']. The text is: "{text.strip()}". The tags must be in the same language as the text. [/INST]"""

    def predict(self, texts):
        inputs = self.tokenizer(
            [self.process_input(text) for text in texts],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        predictions = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            try:
                tags = decoded.split('[/INST]')[-1].strip()
                predictions.append(tags)
            except Exception:
                predictions.append(decoded)
        return predictions

predictor = TagPredictor()

@app.post("/predict", response_model=list[TaggingResponse])
async def predict_tags(request: TaggingRequest):
    try:
        predictions = predictor.predict(request.texts)
        return [TaggingResponse(tags=pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}