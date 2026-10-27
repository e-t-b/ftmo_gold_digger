# Gold Digger — entraînement long PPO, version reprenable

Un script exécutable **train_massive.py**, la politique 5-GRU à huit actions et un checkpoint ont été livrés dans la conversation sous le nom `gold_digger_massive_resume_100eps.zip`.

## Ce qui a effectivement été réalisé

- Reprise à partir du checkpoint intégral : poids de la politique, optimiseur, états aléatoires Python/NumPy/Torch, épisode courant.
- Sauvegarde atomique périodique dans `latest.pt` et suivi `progress.json`.
- Validation par journées entières, distinctes de l'entraînement ; sélection éventuelle de `best_validation.pt` sur cette période uniquement.
- Réinitialisation et reprise vérifiées : le programme a repris après 90 épisodes et a terminé le centième.
- 100 épisodes **effectivement entraînés**, objectif configurable de 10 000 (ce n'est PAS une réalisation déjà effectuée).
- Sur les 62 journées de validation, P&L journalier simulé cumulé : **-$1 863,49** au checkpoint 50, **-$664,85** au checkpoint 100 ; les capitaux sont réinitialisés chaque jour. Ces valeurs ne prouvent aucun edge et ne mesurent pas un challenge FTMO réel.
- Aucune mesure sur la période de test n'a été utilisée pendant l'entraînement de cet essai.

## Comment lancer le vrai entraînement long sur le PC de l'utilisateur

Décompresser l'archive. Ajouter les CSV `m5.csv`, `m15.csv`, `h1.csv`, `h4.csv`, `d1.csv` du serveur FTMO dans `C:\\gold_data` puis :

```powershell
pip install torch numpy pandas
python train_massive.py --input C:\\gold_data --out results --episodes 10000 --save-every 25 --validate-every 250
```

Relancer la même commande après toute interruption ; la sauvegarde reprend à l'épisode suivant sans perdre les poids, optimiseur et RNG. Vérifier `results/progress.json`. La période avril-octobre 2026 ne doit être évaluée qu'à la fin, et non continuellement pour choisir l'architecture.

## Limites importantes

- Les journées tirées à nouveau ne sont pas de nouveaux historiques indépendants.
- Ce simulateur ne reproduit pas encore les fills tick à tick, les commissions et swaps exacts ni le reset de perte flottante FTMO à minuit Prague.
- Les niveaux SL/TP restent calculés, pas entièrement libres ; l'agent LLM discrétionnaire n'est pas encore intégré dans la boucle PPO.
- Ce livrable est une infrastructure d'entraînement extensible, **pas un trader démontré rentable**.

Le code complet et les checkpoints sont fournis comme archive, non publiés dans le dépôt à cette étape.
