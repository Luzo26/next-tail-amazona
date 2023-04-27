# bring in 
from fastapi import FastAPI
from pydantic import BaseModel


from gramformer import Gramformer
import torch

app = FastAPI()

class CorrectingItem(BaseModel):
    sentence: str

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)


gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

def correct(CorrectingItem):
    res = gf.correct(CorrectingItem)
    return res 

@app.get('/')
async def scoring_endpoint():
    return {"Hello":"Dogs, the best thing on earth"}


@app.post('/')
async def scoring_endpoint(item:CorrectingItem):
    res = gf.correct(str(item))
    return res