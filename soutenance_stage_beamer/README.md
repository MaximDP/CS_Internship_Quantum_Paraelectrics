# Soutenance de stage

`main.tex` contient la présentation Beamer et ses références bibliographiques.
Les figures nécessaires sont dans `assets/`.

```sh
sh compile.sh
```

Le PDF final est `soutenance_stage.pdf`. Les fichiers intermédiaires sont
écrits dans `build/`. Une installation TeX Live avec Beamer est nécessaire.
La figure Mayer simplifiée peut être régénérée depuis la racine du dépôt avec
`python soutenance_stage_beamer/scripts/build_mayer_scha_slide_figure.py`.
