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
python -m unittest discover -s tests -v
$env:PYTHONPATH = "src"
python -m sumula_reader schema
```

Dependencias pesadas de visao e ML sao opcionais no bootstrap e serao ativadas nas etapas de alinhamento de PDF e reconhecimento manuscrito.

### Inspecao do template FECABA

```powershell
python -m pip install -e ".[vision]"
scoresheet-parser debug jogo.pdf --output debug/jogo
```

O comando gera a folha normalizada, um overlay das regioes e um crop por
regiao do MVP. Esse artefato e usado para calibrar o template antes da
implementacao dos reconhecedores de pontos e faltas.
