import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

# =========================
# 1. Dados de treino 
# =========================

train_data = {
    "question": [
        "Qual cidade é a capital do Brasil?",
        "Qual cidade é a capital da França?",
        "Qual cidade é a capital da Alemanha?",
        "Qual cidade é a capital dos EUA?",
        "Qual cidade é a capital da Argentina?",
        "Qual é a capital do Canadá?",
        "Qual é a capital de Portugal?",
        "Qual é a capital da Espanha?",
        "Qual é a capital da Itália?",
        "Me diga qual é a capital do Brasil.",
        "Qual é a cidade principal da França?",
        "Qual é a cidade principal da Rússia?",
        "Qual cidade representa o governo da Alemanha?",
        "Qual é a população do Brasil?",
        "Qual é a população da França?",
        "Qual é a população da Alemanha?",
        "Qual é a população dos EUA?",
        "Qual é a população da Argentina?",
        "Qual é a população do Canadá?",
        "Qual é a população de Portugal?",
        "Qual é a população da Espanha?",
        "Qual é a população da Itália?",
        "Quantos estados existem no Brasil?",
        "Quantos estados existem nos EUA?",
        "Quantas províncias existem no Canadá?",
        "Número de habitantes da Alemanha?",
        "Me diga quantas pessoas vivem na França.",
        "Lisboa é a capital de Portugal?",
        "O Brasil tem mais de 30 estados?",
        "A Argentina está na Europa?",
        "Paris é a capital da França?",
        "Berlim é uma cidade do Brasil?",
        "Washington é a capital dos EUA?",
        "Moscou é a capital da Rússia?",
        "Roma fica no Brasil?",
        "Espanha é um país europeu?",
        "Buenos Aires é capital da Argentina?",
        "Quais são os países com mais de 200 milhões de habitantes?",
        "Liste países da Europa citados na base de dados.",
        "Quais capitais são europeias?",
        "Cite os países europeus listados.",
        "Capitais localizadas na Europa?",
        "Qual cidade é a capital da África do Sul?",
        "Qual cidade é a capital da Austrália?",
        "Qual é a capital da China?",
        "Qual é a capital da Índia?",
        "Qual é a capital do Japão?",
        "Qual é a capital do México?",
        "Qual é a capital da Nigéria?",
        "Qual é a capital da Indonésia?",
        "Qual é a cidade principal da Austrália?",
        "Qual é a cidade principal da China?",
        "Qual cidade representa o governo da Índia?",
        "Qual é a população da China?",
        "Qual é a população da Índia?",
        "Qual é a população da Austrália?",
        "Qual é a população do Japão?",
        "Qual é a população do México?",
        "Qual é a população da Nigéria?",
        "Qual é a população da Indonésia?",
        "Número de habitantes da Austrália?",
        "Quantas pessoas vivem na China?",
        "Quantas pessoas vivem na Índia?",
        "Camberra é a capital da Austrália?",
        "A China tem mais de 1 bilhão de habitantes?",
        "A Nigéria está na Europa?",
        "Nova Délhi é a capital da Índia?",
        "Tóquio é uma cidade da China?",
        "Jacarta é a capital da Indonésia?",
        "Pequim é a capital da China?",
        "México é um país asiático?",
        "Nigéria é um país africano?",
        "Buenos Aires é capital da Austrália?",
        "Quais países têm população acima de 1 bilhão?",
        "Liste países asiáticos citados na base de dados.",
        "Quais capitais são asiáticas?",
        "Cite os países da Ásia listados.",
        "Capitais localizadas na Ásia?"
    ],
    "answer": [
        "Brasília", "Paris", "Berlim", "Washington", "Buenos Aires", "Ottawa", "Lisboa", "Madri", "Roma",
        "Brasília", "Paris", "Moscou", "Berlim",
        "213 milhões", "67 milhões", "83 milhões", "331 milhões", "45 milhões", "38 milhões", "10 milhões", "47 milhões", "60 milhões",
        "26 milhões", "50 milhões", "10 milhões", "83 milhões", "67 milhões",
        True, False, False, True, False, True, True, False, True, True,
        ["Brasil", "EUA"],
        ["França", "Alemanha", "Portugal", "Espanha", "Itália"],
        ["Paris", "Berlim", "Lisboa", "Madri", "Roma"],
        ["França", "Alemanha", "Portugal", "Espanha", "Itália"],
        ["Paris", "Berlim", "Lisboa", "Madri", "Roma"],
        "Pretória", "Camberra", "Pequim", "Nova Délhi", "Tóquio", "Cidade do México", "Abuja", "Jacarta",
        "Camberra", "Pequim", "Nova Délhi",
        "1440 milhões", "1410 milhões", "26 milhões", "126 milhões", "128 milhões", "223 milhões", "276 milhões",
        "26 milhões", "1440 milhões", "1410 milhões",
        True, True, False, True, False, True, True, False, True, False,
        ["China", "Índia"],
        ["China", "Índia", "Japão", "Indonésia"],
        ["Pequim", "Nova Délhi", "Tóquio", "Jacarta"],
        ["China", "Índia", "Japão", "Indonésia"],
        ["Pequim", "Nova Délhi", "Tóquio", "Jacarta"]
    ]
}

df_train = pd.DataFrame(train_data)
df_train["answer"] = df_train["answer"].astype(str)

# ======================
# Treinamento do modelo
# ======================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=[
        "a", "à", "ao", "aos", "as", "às", "com", "da", "das", "de", "do", "dos", "e",
        "em", "na", "nas", "no", "nos", "o", "os", "para", "por", "que", "se", "sem",
        "uma", "umas", "um", "uns", "é", "são", "tem", "há", "foi", "ser", "estar"
    ],
    ngram_range=(1, 2),
    strip_accents='unicode'
)

X = vectorizer.fit_transform(df_train["question"])
y = df_train["answer"]
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# =======================
# Questões para predição
# =======================

all_questions = {
    "question": [
        "Qual é a capital da Alemanha?",
        "Qual é a população da França?",
        "A Argentina está na Europa?",
        "Liste populações maiores que 80 milhões.",
        "Qual cidade é a capital da Espanha?",
        "Qual é a capital do Canadá?",
        "Quantos estados tem o Brasil?",
        "Buenos Aires é capital de que país?",
        "Quais capitais estão na Europa?",
        "Quantos habitantes vivem na Alemanha?",
        "Diga três países europeus citados nos dados.",
        "Qual é a capital da Rússia?",
    ]
}
pd.DataFrame(all_questions).to_parquet("all.parquet", index=False)

# ======================
# Função de heurística
# ======================

def heuristica(question, answer):
    q = question.lower()
    a = answer.strip()


    if a.lower() in ["true", "false"]:
        if "é a capital" in q or "está na europa" in q:
            return "Sim" if a.lower() == "true" else "Não"
        return "Não sei"


    if a.startswith("[") and a.endswith("]"):
        a = a.strip("[]").replace("'", "").replace('"', "")
        return ", ".join([x.strip() for x in a.split(",") if x.strip()])


    if "capital de que país" in q:
        capital_to_country = {
            "Buenos Aires": "Argentina",
            "Paris": "França",
            "Brasília": "Brasil",
            "Berlim": "Alemanha",
            "Washington": "Estados Unidos",
            "Ottawa": "Canadá",
            "Lisboa": "Portugal",
            "Madri": "Espanha",
            "Roma": "Itália",
            "Pretória": "África do Sul",
            "Camberra": "Austrália",
            "Pequim": "China",
            "Nova Délhi": "Índia",
            "Tóquio": "Japão",
            "Cidade do México": "México",
            "Abuja": "Nigéria",
            "Jacarta": "Indonésia",
            "Moscou": "Rússia"
        }
        return capital_to_country.get(a, "Não sei")

    return a


# ===========================
# Função para gerar previsões
# ===========================

def predict_from_file(input_parquet_path, output_txt_path):
    df = pd.read_parquet(input_parquet_path)
    if 'question' not in df.columns:
        raise ValueError(f"O arquivo {input_parquet_path} não contém coluna 'question'.")

    questions = df['question'].tolist()
    X_test = vectorizer.transform(questions)
    y_pred = clf.predict(X_test)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for question, answer in zip(questions, y_pred):
            resposta = heuristica(question, answer)
            f.write(f"Pergunta: {question}\nResposta: {resposta}\n\n")

# ===========================
# Execução final
# ===========================

predict_from_file("all.parquet", "predictionsD.txt")
print("Arquivo gerado/atualizado.")
