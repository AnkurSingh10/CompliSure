from pydantic import BaseModel, HttpUrl
class RunRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class RunResponse(BaseModel):
    answers: list[str]
