from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# =========================
# CORS (OBLIGATORIO PARA REACT NATIVE / FRONTEND)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción puedes limitar
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CARGAR MODELO
# =========================
modelo = joblib.load("model.pkl")

# =========================
# FEATURES (MISMO ORDEN QUE ENTRENAMIENTO)
# =========================
features = [
    'Drawing','Dancing','Singing','Sports','Video Game','Acting','Travelling',
    'Gardening','Animals','Photography','Teaching','Exercise','Coding',
    'Electricity Components','Mechanic Parts','Computer Parts','Researching',
    'Architecture','Historic Collection','Botany','Zoology','Physics',
    'Accounting','Economics','Sociology','Geography','Psycology','History',
    'Science','Bussiness Education','Chemistry','Mathematics','Biology',
    'Makeup','Designing','Content writing','Crafting','Literature','Reading',
    'Cartooning','Debating','Asrtology','Hindi','French','English','Urdu',
    'Other Language','Solving Puzzles','Gymnastics','Yoga','Engeeniering',
    'Doctor','Pharmisist','Cycling','Knitting','Director','Journalism',
    'Bussiness','Listening Music'
]

# =========================
# MODELO DE ENTRADA (IMPORTANTE PARA REACT NATIVE)
# =========================
class InputData(BaseModel):
    Drawing: int = 0
    Dancing: int = 0
    Singing: int = 0
    Sports: int = 0
    Video_Game: int = 0
    Acting: int = 0
    Travelling: int = 0
    Gardening: int = 0
    Animals: int = 0
    Photography: int = 0
    Teaching: int = 0
    Exercise: int = 0
    Coding: int = 0
    Electricity_Components: int = 0
    Mechanic_Parts: int = 0
    Computer_Parts: int = 0
    Researching: int = 0
    Architecture: int = 0
    Historic_Collection: int = 0
    Botany: int = 0
    Zoology: int = 0
    Physics: int = 0
    Accounting: int = 0
    Economics: int = 0
    Sociology: int = 0
    Geography: int = 0
    Psycology: int = 0
    History: int = 0
    Science: int = 0
    Bussiness_Education: int = 0
    Chemistry: int = 0
    Mathematics: int = 0
    Biology: int = 0
    Makeup: int = 0
    Designing: int = 0
    Content_writing: int = 0
    Crafting: int = 0
    Literature: int = 0
    Reading: int = 0
    Cartooning: int = 0
    Debating: int = 0
    Asrtology: int = 0
    Hindi: int = 0
    French: int = 0
    English: int = 0
    Urdu: int = 0
    Other_Language: int = 0
    Solving_Puzzles: int = 0
    Gymnastics: int = 0
    Yoga: int = 0
    Engeeniering: int = 0
    Doctor: int = 0
    Pharmisist: int = 0
    Cycling: int = 0
    Knitting: int = 0
    Director: int = 0
    Journalism: int = 0
    Bussiness: int = 0
    Listening_Music: int = 0

# =========================
# ENDPOINT
# =========================
@app.post("/predict")
def predict(data: InputData):

    values = [
        data.Drawing, data.Dancing, data.Singing, data.Sports, data.Video_Game,
        data.Acting, data.Travelling, data.Gardening, data.Animals,
        data.Photography, data.Teaching, data.Exercise, data.Coding,
        data.Electricity_Components, data.Mechanic_Parts, data.Computer_Parts,
        data.Researching, data.Architecture, data.Historic_Collection,
        data.Botany, data.Zoology, data.Physics, data.Accounting,
        data.Economics, data.Sociology, data.Geography, data.Psycology,
        data.History, data.Science, data.Bussiness_Education, data.Chemistry,
        data.Mathematics, data.Biology, data.Makeup, data.Designing,
        data.Content_writing, data.Crafting, data.Literature, data.Reading,
        data.Cartooning, data.Debating, data.Asrtology, data.Hindi,
        data.French, data.English, data.Urdu, data.Other_Language,
        data.Solving_Puzzles, data.Gymnastics, data.Yoga, data.Engeeniering,
        data.Doctor, data.Pharmisist, data.Cycling, data.Knitting,
        data.Director, data.Journalism, data.Bussiness, data.Listening_Music
    ]

    X = np.array(values).reshape(1, -1)

    pred = modelo.predict(X)

    return {
        "resultado": str(pred[0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)