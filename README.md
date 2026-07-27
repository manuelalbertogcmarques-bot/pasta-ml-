# Modelo XGBoost - Previsão de Duração de Instalações

## Setup

1. Criar ambiente virtual e instalar dependências:

```bash
cd ml
python -m venv venv
source venv/bin/activate    # Linux/Mac
# ou: venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Treinar o modelo (lê os dados do Supabase e guarda `model.json`):

```bash
python train_model.py
```

3. Arrancar a API:

```bash
uvicorn app:app --reload --port 8000
```

A API vai estar disponível em `http://localhost:8000`.

## Endpoints

- `POST /prever` — recebe `{"m_cabo": 30, "tecnico": "tecnico1"}` e devolve `{"tempo_previsto": 88.5, "m_cabo": 30, "tecnico": "tecnico1"}`
- `GET /health` — verifica se a API está a correr e o modelo carregado

## Retreinar

Sempre que houver novos dados na tabela `instalacoes`, voltar a correr `python train_model.py` e reiniciar a API.
