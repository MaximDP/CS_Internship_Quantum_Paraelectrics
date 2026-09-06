# Entrées et provenance

`examples/batio3_500K.json` configure le point d’entrée `run_pipeline.py`.
Les données numériques ci-dessous sont lues par les scripts de comparaison ;
leurs colonnes nomment explicitement les unités.

| Fichier | Origine et usage |
|---|---|
| `paraelectric/wieczorek2006_batio3_fig1a_digitized.csv` | 14 points numérisés de la figure 1(a), permittivité de BaTiO₃ à 1 kHz ; comparaison et calibration inverse. |
| `paraelectric/nakatani2016_batio3_cubic_lattice.csv` | Paramètre de maille cubique et incertitudes ; ajustement principal limité à 413–598 K. |
| `paraelectric/bst_curie_weiss_reference.csv` | Cibles nominales reconstruites pour x = 0,4 ; 0,5 ; 0,6 d’après Weerasinghe et al. (2013), 10 kHz. Les incertitudes sont conservées. |
| `mayer/mayer2022_table_I.csv` | Paramètres conventionnels PBEsol de Mayer et al., Table I. |
| `mayer/mayer2022_anharmonic_reduction.csv` | Coefficients k1 et k4 après élimination des modes optiques secondaires. |
| `mayer/mayer2022_batio3_fig3a_experiment_digitized.csv` | Points expérimentaux numérisés dans la figure 3(a), 302–470 K ; aucune extrapolation de ces points. |
| `mayer/nakatani2016_batio3_cubic_lattice.csv` | Même référence de maille pour le protocole Mayer. |
| `mayer/wieczorek2006_batio3_fig1a_digitized.csv` | Même permittivité mesurée ; la colonne de susceptibilité historique n’est pas utilisée pour Mayer, qui recalcule avec epsilon_inf = 6,847. |

Les tables des paramètres sont documentaires. Les valeurs utilisées par les
solveurs sont définies dans `SCHA_paraelc/scha_paraelc.py`,
`tenseur_beta/constants.py`, `tenseur_C/constants.py`, `lattice_sum/materials.py`
et `Mayer/parameters.py`. Les noms de matériaux et conversions doivent rester
cohérents entre ces modules. Les références complètes et DOI figurent dans
`rapportCS_stage_maxime/biblio.bib` à la racine du dépôt.

Les courbes Nishimatsu recalculées sont conservées sous
`outputs/reference/mayer/`, car ce sont des sorties théoriques.
