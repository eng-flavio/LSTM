# Uilizando LSTM para previsão de temperatura
Experimento que usa séries temporais para previsão de amostras futuras de um determinado atributo contido em um *dataset* gerado pelo Instituto Max Planck de Biogeoquímica. Para tanto, foram utilizados as Redes Neurais Recorrentes (RNN)  Long Short Term Memory (LSTM).

## Requisitos
O experimento requer a versão  `<=3.10` do python. Todos os testes foram realizados no Sistema Operacional Windows 11. Recomenda-se a criação de um ambiente à parte.

Para ter certeza que todos os requisitos necessários para o experimento estão instalados, execute o seguinte  comando no seu ambiente no diretório principal do projeto

**shell**:

`pip install -r requirements.txt`

## Arquivo `main_ntb.ipynb`
O arquivo em questão referente à criação de um notebook para previsão de séries temporais. Para dar início à apresentação, basta acessar o arquivo `main_ntb.ipynb`. 

## Arquivo mlflowscript.py
Esse arquivo visa o uso da ferramenta *mlflow* para realizar o mesmo tipo de processamento.  Para executar o arquivo, execute o seguinte comando no terminal no diretório da questão:

**shell**:

`python questaobonus.py`

Você também pode adicionar e modificar os seguintes parâmetros e hiperparâmetros para melhorar a performance do aprendizado:

**shell**:

`python questaobonus.py --n_future 20 --epochs 100 --neurons 13 --offset 2 --look_back 5`

Os valores acima são padrões e opcionais, mas podem ser modificados, cada um desses parâmetros ou hiperparâmetros representam:

- `n_future`: quantidade de amostras futuras que serão previstas

- `epochs`: quantidade de épocas de treinamento

- `neurons`: quantidade de neurônios das duas primeiras camadas da LSTM

- `offset`: *offset* de amostras anteriores que não serão utilizadas na previsão

- `look_back`: quantidade de amostras anteriores utilizadas para previsão das amostras futuras

Alguns atributos foram fixados, por exemplo o `batch_size`, mas podem facilmente serem alterados no arquivo `questaobonus.py`.

Após o treinamento ser sucedido clique no *link* gerado no `terminal` ou acesse *https://my-server:5000* para visualizar os resultados gerados no `mlflow`.
