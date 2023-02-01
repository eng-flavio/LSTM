# PROCESSO DE SELEÇÃO PARA O CARGO DE CIENTISTA DE DADOS I
Experimento realizado na seleção para o cargo de Cientista de Dados I na Federação das Indústrias do Estado do Ceará (FIEC). Esse experimento exige que o candidato utilize séries temporais para previsão de amostras futuras de um determinado atributo contido em um *dataset* gerado pelo Instituto Max Planck de Biogeoquímica. Para tanto, foram utilizados as Redes Neurais Recorrentes(RNN)  Long Short Term Memory(LSTM).

## Requisitos
O experimento requer a versão  `3.9.16` do python. Todos os testes foram realizados no Sistema Operacional Windows 11.

Para ter certeza que todos os requisitos necessários para o experimento estão instalados, execute o seguinte  comando no seu ambiente no diretório principal do projeto

**shell**:

`pip install -r requirements.txt`

Para o sistema Linux também é necessário instalar a seguinte dependência:

**shell**:

`sudo apt-get install libsnappy-dev`

## Questão 1
A questão 1 é referente à criação de um notebook para previsão de séries temporais. Para dar início à apresentação, basta executar o arquivo `.ipynb`. 

## Questão 2
Essa questão visa o uso da ferramenta *mlflow* para realizar o mesmo tipo de processamento.  Para executar a questão bônus, execute o seguinte comando no terminal no diretório da questão:

**shell**:

`python questaobonus.py`

Você também pode adicionar e modificar os seguintes parâmetros e hiperparâmetros para melhorar a performance do aprendizado:

**shell**:

`python questaobonus.py --n_future 20 --epochs 100 --neurons 13 --offset 2 --look_back 5`

Os valores acima são padrões e opcionais, mas podem ser modificados, cada um desses parâmetros ou hiperparâmetros representam :

- `n_future`: quantidade de amostras futuras que serão previstas

- `epochs`: quantidade de épocas de treinamento

- `neurons`: quantidade de neurônios das duas primeiras camadas da LSTM

- `offset`: *offset* de amostras anteriores que não serão utilizadas na previsão

- `look_back`: quantidade de amostras anteriores utilizadas para previsão das amostras futuras

Após o treinamento ser sucedido clique no *link* gerado no `terminal` ou acesse *https://my-server:5000* para visualizar os resultados gerados no `mlflow`.
