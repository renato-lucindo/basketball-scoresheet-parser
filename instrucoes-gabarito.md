# Instruções para gerar o gabarito da súmula

Analise visualmente a súmula de basquete e gere um gabarito JSON fiel ao documento. O objetivo é usar esse gabarito para treinar um sistema de reconhecimento manuscrito.

Não corrija o que o apontador escreveu. Registre o que estiver realmente marcado na súmula. Se alguma informação estiver ilegível ou duvidosa, não adivinhe: use `null` e registre a dúvida em `warnings`.

Para cada equipe, identifique:

1. **Elenco**: apenas os números das camisas.
2. **Pontuação**: evento por evento, na ordem em que aparece, contendo:
   - quarto (`Q1` a `Q4`);
   - número da camisa;
   - valor da cesta: `1`, `2` ou `3`;
   - placar acumulado da própria equipe imediatamente após aquela cesta.
3. **Faltas coletivas por quarto**:
   - conte apenas as caixas marcadas com `X`;
   - traços horizontais usados para fechar ou inutilizar caixas não contam como faltas.
4. **Faltas individuais**, preservando a ordem:
   - número da camisa;
   - quarto;
   - símbolo exatamente como escrito, por exemplo `P`, `P1`, `P2`, `P3`, `T`, `T1`, `U`, `U1`, `U2`, `D`, `GD`.
5. **Apontador**: se o campo estiver legível, preencha `writer_name`. Se não estiver, use `null`.

## Regras importantes

- Não inferir uma camisa apenas com base em quem normalmente joga.
- Não completar uma falta que pareça provável.
- Não transformar `GD` em `D` ou em outro símbolo.
- Preserve faltas repetidas, por exemplo `['T1', 'T1', 'GD']`.
- Uma cesta de 3 pontos é aquela em que a marcação de cesta está acompanhada do número do jogador circulado.
- Lance livre vale 1 ponto e normalmente é indicado por um ponto sobre o número da progressão.
- Cesta de 2 pontos vale 2 e normalmente usa traço diagonal.
- Caso haja rasura, registre o valor final somente se ele estiver claro; caso contrário use `null`.
- O placar e as parciais devem ser conferidos pela soma dos eventos, mas não altere eventos apenas para fazer a soma bater.
- Retorne somente o JSON final, sem explicações adicionais.
