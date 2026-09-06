# Rapport de stage

`rapportCS.tex` inclut les chapitres français, l’annexe des paramètres et
l’annexe ferroélectrique. `biblio.bib` rassemble les références ; `images/`,
`figures/` et `logos/` contiennent les ressources utilisées par le document.

```sh
sh compile.sh
```

Le script utilise `pdflatex`, `bibtex`, puis deux passes de `pdflatex`.
Le PDF est écrit dans `output/pdf/rapportCS.pdf` et les auxiliaires dans `build/`.
La classe institutionnelle `rapportCS.cls` est incluse. Une distribution TeX
Live complète est recommandée.
