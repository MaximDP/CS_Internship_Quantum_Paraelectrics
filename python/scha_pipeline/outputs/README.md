# Sorties

`reference/` contient les tables et figures numériques livrées avec le rapport.
Les nouveaux calculs sont écrits dans `example/`, `paraelectric/`,
`ferroelectric/` ou `mayer/`, selon le script. Ces répertoires de travail sont
ignorés par Git. Pour conserver un nouveau calcul, lui donner un nom explicite
et enregistrer ensemble sa configuration, ses tables et ses figures.

- CSV : résultats par température et protocole ; unités dans les en-têtes.
- JSON : paramètres, choix de calcul et diagnostics de la solution.
- PNG : visualisation des tables numériques.

Les masses `A1` sont exprimées dans la convention du déplacement local u.
Les réponses `chi_soft` sont en unités gaussiennes. Les valeurs non finies des
scans signalent une branche absente ou instable ; elles ne représentent pas
une susceptibilité nulle. Les quantités post-SCHA sont des diagnostics à une
insertion, au-delà du domaine perturbatif dans ces résultats.
