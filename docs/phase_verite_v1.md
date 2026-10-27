# Gold Digger — Phase de vérité V1, 2026-10-27

## Verdict réel : NO-GO sur le GRU V3 actuel

Replay chronologique d'un checkpoint **PyTorch GRU 5TF déjà entraîné**
sur XAUUSD FTMO-Demo. La fenêtre diagnostique du **2 avril au 27 octobre 2026**
a déjà été inspectée lors des travaux précédents : **ce n'est plus un vrai
holdout inédit**.

| Protocole | Trades | Win rate | Profit factor | Net sur 10 000 $ | Drawdown du solde réalisé |
|---|---:|---:|---:|---:|---:|
| GRU, spread M5 enregistré | 430 | 47,44 % | 0,874 | **-620,55 $** | 9,26 % |
| GRU, spread M5 ×2 | 430 | 43,26 % | 0,754 | **-1 235,77 $** | 12,63 % |
| Référence naïve achat systématique | 496 | 40,73 % | 0,656 | **-1 973,30 $** | 20,76 % |

Les chiffres sont issus des 9 299 snapshots de la partition diagnostique,
avec prédictions recomputées par le checkpoint gelé (pas de réentraînement).
L'agent discrétionnaire/LLM **n'a pas été utilisé dans ce replay**.

## Décision de trading simulée

- Applique la normalisation stockée avec le checkpoint au dernier contexte
  de bougies **clôturées** M5/M15/H1/H4/D1.
- Choisit le score maximum parmi DOWN/WAIT/UP. Propose une direction
  seulement si DOWN ou UP obtient ≥0,50 ; sinon s'abstient.
- Exécute à l'ouverture de la bougie M5 **suivante** (pas de lookahead).
  BUY à Ask ≈ Bid + spread M5 ; SELL au Bid ; glissement adverse hypothétique
  de **0,05 $/oz de chaque côté**.
- SL = TP = max(1 $, 1,25 × ATR14 des bougies passées, 1,23 $).
  Horizon maximum 12 M5 ; stop avant cible si les deux sont touchés dans
  la même bougie. Aucun maintien délibéré à travers une clôture de marché.
- Capital initial 10 000 $, risque nominal 0,25 % par position,
  pas minimum 0,01 lot (1 oz), max 3 trades/jour, arrêt interne sur
  pertes **réalisées** de 1 %/jour, changement de date Europe/Prague.
- Répète à spread double comme scénario de sensibilité ; référence
  naive BUY sur les points d'observation et les mêmes règles de taille.

## Limites — NE PAS INTERPRÉTER COMME BACKTEST FTMO EXACT

1. Bid OHLC + « spread par bougie » ne reproduit PAS les quotes Ask/Bid
   tick par tick au moment réel du stop, de l'entrée ou de la sortie.
2. Glissement fixe et zéro commission sont des hypothèses ; swaps,
   requotes, liquidité et marge non reproduits.
3. Le suivi FTMO réel se fait sur equity **y compris positions flottantes**
   et frais. Ici, seulement le solde réalisé : aucun certificat de
   respect des règles FTMO, aucune probabilité de réussite 2-Step.
4. Drawdown affiché sur solde **réalisé**, pas l'equity intratrade.
5. Politique mécanique 1R basée sur la sortie du GRU : ce n'est PAS
   encore l'agent IA décisionnaire libre, ni un backtest de ce dernier.
6. La période diagnostique a déjà été utilisée par plusieurs expériences :
   ne pas l'appeler « untouched out-of-sample ».

## Fichiers de reproduction

L'archive livrée dans la conversation,
`gold_digger_phase_verite_v1.zip`, contient :

- `truth_backtest.py` : replay, prix d'entrée/sortie, P&L et contrôle des
  décisions ; script exécuté sur CSV historiques FTMO fournis ;
- `test_truth_backtest.py` : **3 tests ciblés exécutés avec succès** ;
- `train_gold_barrier_v3.py`, `train_gold_direction_v2.py` : définitions
  exactes des features et de l'architecture ;
- `gold_barrier_v3/model.pt` : poids réellement entraînés précédemment ;
- `result/summary.json` : hypothèses et résultats chiffrés ;
- `result/*_trades.csv` et `*_equity.csv` : détails de chaque trade et
  évolution du capital, par scénario ;
- `result/REPORT_FR.md` : rapport francophone.

Exécution depuis le dossier de l'archive décompressée (nécessite Torch,
Pandas, NumPy, scikit-learn, et les CSV historiques dans `--input`) :

```powershell
python truth_backtest.py --input data/gold_3y --checkpoint gold_barrier_v3/model.pt --out result
pytest -q test_truth_backtest.py
```

**Décision : NO-GO.** Travail suivant : replay du véritable agent
discrétionnaire, états de compte complets, risque FTMO en floating equity,
et nouvelle période de forward test indépendante.
