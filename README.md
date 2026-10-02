# basketball-scoresheet-parser

Leitor estruturado de sumulas de basquete, inicialmente focado no modelo FECABA.

O MVP extrai e valida:

- equipes e numeros de camisa;
- jogadores que entraram em quadra;
- quinteto inicial;
- apontador;
- eventos de pontuacao por periodo;
- faltas individuais e faltas da equipe;
- parciais e placar final para validacao.

## Regras de participacao

Na coluna de entrada em quadra:

- sem `X`: nao participou;
- `X` azul: participou;
- `X` vermelho: participou;
- `X` azul circulado de vermelho: participou e foi titular.

O leitor mantem observacao, interpretacao, confianca e status separados. Dados ambiguos devem ser enviados para revisao em vez de serem corrigidos silenciosamente.

## Desenvolvimento

```powershell
python -m pytest -q
python -m sumula_reader schema
```

Dependencias pesadas de visao e ML sao opcionais no bootstrap e usadas nas etapas de alinhamento de PDF e reconhecimento manuscrito.

### Baseline de digitos manuscritos

O primeiro marco de ML pre-treina uma CNN em `EMNIST Digits`. O dataset e os
checkpoints ficam fora do Git.

```powershell
python -m pip install -e ".[ml]"
python -m sumula_reader train-digits --epochs 5
```

Para validar rapidamente o ambiente antes do treino completo:

```powershell
python -m sumula_reader train-digits `
  --epochs 1 `
  --max-train-samples 1024 `
  --max-test-samples 512 `
  --output models/digit-cnn-smoke.pt
```

O checkpoint inclui os pesos, normalizacao, configuracao de treino, acuracia
global e acuracia por digito. Ele serve como baseline para os reconhecedores
de camisas e simbolos manuscritos.

### Dataset FECABA e reconhecimento manuscrito

`dataset-ingest` extrai e cataloga as sumulas do ZIP. `dataset-build` gera
somente candidatos ancorados em grades detectadas: eventos de pontuacao combinam
camisa + marcacao, e faltas individuais usam as cinco celulas reais da linha.
Cada registro de `manifest.jsonl` nasce com `review_state=pending`.

```powershell
python -m sumula_reader dataset-ingest sumulas.zip --output datasets/fecaba --pilot-count 5
python -m sumula_reader dataset-build --dataset-root datasets/fecaba
```

### Revisao assistida com Label Studio

A revisao e local e usa duas passagens: primeiro as caixas das cinco sumulas
piloto; depois os candidatos de pontuacao e faltas. As previsoes ficam no JSON de
tarefas e o manifesto automatico continua separado do manifesto revisado.

```powershell
python -m pip install -e ".[vision,review]"
python -m sumula_reader dataset-review-prepare --stage geometry

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=(Resolve-Path "datasets/fecaba").Path
label-studio start
```

No Label Studio, crie um projeto, use o conteudo de
`datasets/fecaba/review/geometry.labeling.xml` como labeling config e importe
`geometry.tasks.json`. As caixas chegam em `predictions`; copie a previsao para
uma anotacao antes de editar. Depois de exportar JSON:

```powershell
python -m sumula_reader dataset-review-import geometry-export.json --stage geometry
python -m sumula_reader dataset-build --dataset-root datasets/fecaba
python -m sumula_reader dataset-review-prepare --stage scoring
python -m sumula_reader dataset-review-prepare --stage fouls
```

Para cada crop, escolha `accepted`, `adjusted`, uma das rejeicoes, `erasure` ou
`illegible`, ajuste a caixa quando necessario e informe o rotulo final. Importe
os dois exports e confira a cobertura:

```powershell
python -m sumula_reader dataset-review-import scoring-export.json --stage scoring
python -m sumula_reader dataset-review-import fouls-export.json --stage fouls
python -m sumula_reader dataset-review-status
```

Depois da primeira revisao completa, o import marca exatamente 10% dos crops
treinaveis (arredondando para cima) para uma segunda conferencia. Gere e
importe essa amostra separadamente:

```powershell
python -m sumula_reader dataset-review-prepare --stage scoring --audit-only
python -m sumula_reader dataset-review-import scoring-audit-export.json --stage scoring --audit-only
python -m sumula_reader dataset-review-prepare --stage fouls --audit-only
python -m sumula_reader dataset-review-import fouls-audit-export.json --stage fouls --audit-only
python -m sumula_reader dataset-review-status
```

O status exige 100% de decisoes, nenhuma revisao obsoleta, auditoria concluida
e pelo menos 98% de cortes geometricamente corretos antes de marcar o lote como
pronto para treinamento. Ao reexecutar `dataset-build`, o manifesto automatico
continua preservado; o resumo soma os registros revisados ainda validos em
`reviewed_labeled`, permitindo confirmar `labeled > 0` sem copiar rotulos para
o manifesto automatico.

O resultado fica em `datasets/fecaba/crops/manifest.reviewed.jsonl`. Cada
registro preserva `crop_id`, documento/apontador, hash da sumula, caixa
original/revisada, rotulo original/final, equipe, periodo, posicao e estado da
revisao. Apenas `accepted` e `adjusted` com rotulo valido entram no treinamento.
Camisas sao validadas em `0..99`; faltas usam o vocabulario do reconhecedor.
`foul_terminal` registra `F`/traco de encerramento separadamente e nunca entra
no classificador de faltas comuns.
Caixas fora da imagem, IDs duplicados e exports de uma versao antiga da sumula
sao rejeitados.

O split e feito por documento. Quando `writer_name` estiver preenchido no
gabarito, o apontador anonimizado passa a ser a unidade de agrupamento, para
que a mesma caligrafia nao apareca em treino e teste. O build valida esse
isolamento antes de gravar o manifesto.

```powershell
python -m sumula_reader train-jerseys `
  --manifest datasets/fecaba/crops/manifest.reviewed.jsonl `
  --output models/handwriting/jersey.pt

python -m sumula_reader train-fouls `
  --manifest datasets/fecaba/crops/manifest.reviewed.jsonl `
  --output models/handwriting/foul.pt
```

Na baseline atual, o reconhecedor de camisas atingiu 90,95% no holdout
sintetico, com 65,5% de automacao no limiar calibrado e erro aceito abaixo de
1%. O classificador de faltas atingiu 95,7%, 90,7% de automacao e erro aceito
abaixo de 1%. Essas metricas ainda nao medem generalizacao real: os cinco
gabaritos piloto continuam pendentes de rotulacao.

### Inspecao do template FECABA

```powershell
python -m pip install -e ".[vision]"
scoresheet-parser debug jogo.pdf --output debug/jogo
```

O comando gera a folha normalizada, um overlay das regioes e um crop por
regiao do MVP. Esse artefato e usado para calibrar o template antes da
implementacao dos reconhecedores de pontos e faltas.

## Estado atual do MVP

- geometria e normalizacao do template FECABA;
- participacao e quinteto inicial por cor/marca;
- pontuacao FIBA 2024: quatro paineis, linhas 1-160 e pares camisa + marcacao;
- vermelho como sinal auxiliar de Q1/Q3 e azul de Q2/Q4;
- faltas da equipe: X, casa inutilizada e limite de quatro marcas;
- faltas individuais: cinco celulas reais, terminais separados e separador do intervalo;
- reconhecedor PyTorch de camisas de uma ou duas casas;
- reconhecedor PyTorch de `P`, `P1-P3`, `T`, `T1`, `U`, `U1-U3`, `D`, `D2`, `GD`, `Pc`, `Tc`, `Uc` e `Dc`;
- calibracao de confianca para manter previsoes duvidosas em revisao.

O reconhecimento manuscrito fica atras de interfaces proprias e pode ser
ativado por checkpoint. Previsoes abaixo do limiar calibrado continuam
marcadas para revisao.

### Analise estruturada

O roster e fornecido como contexto para restringir as previsoes de camisa:

    scoresheet-parser analyze jogo.pdf ^
      --roster-a "4,5,6,7,8,9,10,11,12,13,14" ^
      --roster-b "4,5,6,7,8,11,12,13,14,15,16,17" ^
      --writer-id writer_07 ^
      --handwriting-model-dir models/handwriting ^
      --output resultado.json

O JSON contem evidencias visuais, periodos, placar calculado, faltas,
participacao, previsoes manuscritas e warnings. Casos abaixo do limiar de
confianca permanecem em revisao.

Uma regressao real de referencia para Amprl x Lusb u15M preserva placar
38 x 76, parciais 8/13/4/13 e 16/26/13/21, e faltas coletivas da Equipe A
4/4/2/4. Os dois tracos horizontais usados para inutilizar uma caixa de falta
coletiva prevalecem sobre um X anterior/rasurado quando a geometria horizontal
e clara.

## Jev / TypeSafe (opcional)

O projeto possui um cliente opcional para a API TypeSafe em
`sumula_reader.jev`. A chave nunca deve ser salva no repositorio. O comando
de analise solicita a chave de forma interativa quando `TYPESAFE_API_KEY`
nao estiver definida.

Para validar a chave sem grava-la em arquivo ou no historico do terminal:

```powershell
python .\src\sumula_reader\jev.py
```

O comando solicita a chave de forma interativa e lista os modelos liberados
para a conta. Para usar Jev durante a analise:

```powershell
python -m src.sumula_reader.cli analyze jogo.pdf `
  --roster-a "4,5,6,7,8" `
  --roster-b "9,10,11,12,13" `
  --jev
```

O modelo padrao e `jev-latest`. Para testar o alias de preview disponivel na
conta, acrescente `--jev-model jev-preview`. O Jev e consultado somente nos
casos que a classificacao visual local ja marcou como ambigua; leituras claras
continuam locais.

Para instalar as dependencias dessa integracao em outro ambiente:

```powershell
python -m pip install -e ".[jev]"
```

O `JevDecisionEngine` recebe apenas evidencias estruturadas produzidas pela
visao (por exemplo, densidade de tinta, orientacao de tracos, periodo e slot)
e pode classificar casos ambiguos de falta coletiva ou marcacao de pontuacao.
Ele nao substitui o reconhecimento visual/manuscrito.
