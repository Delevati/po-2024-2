# Lista 01 - Pesquisa Operacional

### Observação

run code e resultados carregados em: ```lista01.ipynb```

O requirements desse projeto foi criado com `pip` (tipo `venv` ou `virtualenv`). Não é compátivel com `conda`.

## Conteúdo do Notebook

### Questão I
- **Cell 1**: All imports

- **Cell 2**: Implementação do problema de Bin Packing usando Simulated Annealing (SA). Pega itens de diferentes tamanhos e tenta alocá-los no menor número possível de bins.

### Questão II
- **Cell 4**: Problema das Formas - Análise e agrupamento espacial de cidades usando algoritmo guloso (GDA) e busca local guiada (GLS). Foi feita com dados reais do estado de Alagoas, população e localizações. Na GLS, Maceió ficou fora porque o algoritmo levou em consideração a população para definir a região.

- **Cell 5**: Problema do Plantio resolvido com Algoritmo Genético (AG). Usei dados como de consumo de água da cultivar por m³ e valores por hectare das culturas, com uma pequena normalização dos valores para equilibrar as decisões e permitir uma visualização mais rica dos resultados. Também incluí penalizações para monoculturas, forçando a presença de pelo menos duas culturas diferentes em cada fazenda.

- **Cell 6**: Problema da ração animal resolvido usando Programação Linear. Implementei tanto o método gráfico (visual) quanto o PuLP para resolver e comparar resultados.

- **Cell 7**: Problema do Clique Máximo com três abordagens diferentes: Programação Linear Inteira (PLI), algoritmo guloso e GRASP. Legal ver como cada método se comporta.

- **Cell 8**: Problema de Rota resolvido com o algoritmo A* para encontrar o melhor caminho entre cidades do estado de Alagoas. Implementei uma heurística manual que considera a distância euclidiana.

Cada célula pode conter mais de um método para buscar o espaço de solução ótima, permitindo comparações entre diferentes abordagens.