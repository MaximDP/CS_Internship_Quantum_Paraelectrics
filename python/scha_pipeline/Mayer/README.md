# BaTiO₃ : paramètres Mayer et diagnostic post-SCHA

Cette branche utilise le jeu PBEsol de Mayer et al., *Physical Review B* 106,
064108 (2022), avec réduction anharmonique des modes optiques secondaires.
`parameters.py` conserve les paramètres conventionnels et effectifs, ainsi
que les conversions vers les unités atomiques.

Les entrées sont dans `../inputs/mayer/`, les tables livrées dans
`../outputs/reference/mayer/` et les nouveaux calculs dans `../outputs/mayer/`.
Les [instructions de la pipeline](../README.md) donnent les commandes complètes.

- `run_mayer_bto.py` : scans à pression nulle, à maille ajustée point par point
  et avec approximation affine de la pression.
- `build_mayer_chi_figure.py` : comparaison des tables livrées avec les points
  numérisés de Mayer et Wieczorek.
- `compute_a1_transition_table.py` : limites A1 → 0⁺ de la branche paraélectrique,
  distinctes d’une température de coexistence.
- `post_scha_quartic.py` : sunset local projeté à deux vertex quartiques habillés.
- `test_mayer.py` : conversions, symétries et propriétés numériques.

La pression de maille satisfait `a_SCHA[T,p_a(T)] = a_exp_fit(T)`. Le fit de
maille utilise 413–598 K ; son prolongement à 800 K est extrapolé. La masse
acoustique de 46,44 amu est la valeur des courbes livrées, distincte de la masse polaire et des
paramètres ajustés de Mayer. Nishimatsu 2016, Table I, indique 46,64 amu.
À 500 K, pression nulle et grille 40³, utiliser 46,64 change la susceptibilité
de moins de 0,0003 % pour les jeux BaTiO₃ Nishimatsu et Mayer.

Les points Wieczorek et Mayer ne calibrent pas cette pression. La conversion
expérimentale utilise `chi_soft = (epsilon_r - 6.847)/(4*pi)`.
Le sunset inclut les contractions locales sextiques et octiques, mais omet
les vertex acoustiques dynamiques et les autres squelettes à deux insertions.
Le rapport de correction supérieur à l’unité exclut une interprétation comme
prédiction perturbative convergée.

La courbure en Gamma est ancrée à `2*kappa = -0.0404430732 Ha/bohr²`.
La reconstruction directe des autres paramètres publiés donne
`-0.0412428397 Ha/bohr²` dans la convention Ewald utilisée. Le noyau de calcul
soustrait cette dernière valeur puis ajoute la masse A1 : ce choix représente
un décalage harmonique nu de `+0.0007997665 Ha/bohr²`. L’écart dépasse la seule
précision d’affichage des paramètres. Il limite l’identification exacte du
modèle numérique à celui de la dynamique moléculaire publiée.
