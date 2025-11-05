# Avaliador de Ações

Aplicativo com filtros customizáveis para análise de ações.

## Dependências

- Conta ativa no Interactive Brokers
- Python 3.11+
- Requests, Pandas, NumPy, API do Yahoo Finance, API do Interactive Brokers

# Instalação de Dependências

```bash
pip install ib_insync pandas numpy yfinance requests
```

## Instalação

```bash
git clone https://github.com/felipe-lisandro/avaliador-de-acoes
cd avaliador-de-acoes
python main.py
```

## Funcionalidades

- Criação de filtros personalizados para análise de ações
- Adição e Remoção de ações
- Análise de ações no banco de dados local
- Obtenção de dados de mercado (necessita conta Interactive Brokers) e dados fundamentais
- Display de melhores ações dentro do filtro
