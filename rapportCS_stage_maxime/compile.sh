#!/bin/sh

set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir="$project_dir/build"
output_dir="$project_dir/output/pdf"

# TeX Live 2026 Basic ne contient pas tous les paquets utilises par la classe
# institutionnelle. Sur la machine de travail, ils sont encore disponibles
# dans l'installation TeX Live 2025. Le chemin standard reste prioritaire et
# ce complement n'est ajoute que s'il existe.
primary_tex=/usr/local/texlive/2026basic/texmf-dist/tex//
fallback_tex=/usr/local/texlive/2025/texmf-dist/tex//
primary_bst=/usr/local/texlive/2026basic/texmf-dist/bibtex/bst//
fallback_bst=/usr/local/texlive/2025/texmf-dist/bibtex/bst//

tex_search_path="$project_dir:."
if [ -d "$primary_tex" ]; then
    tex_search_path="$tex_search_path:$primary_tex"
fi
if [ -d "$fallback_tex" ]; then
    tex_search_path="$tex_search_path:$fallback_tex"
fi
export TEXINPUTS="$tex_search_path:${TEXINPUTS:-}"

# BibTeX est lancé depuis build/. Le répertoire courant doit rester prioritaire
# afin qu'il lise l'auxiliaire fraîchement produit, avant toute ancienne copie
# éventuellement présente à la racine du projet.
export BIBINPUTS=".:$project_dir${BIBINPUTS:+:$BIBINPUTS}"
bst_search_path=""
if [ -d "$primary_bst" ]; then
    bst_search_path="$primary_bst"
fi
if [ -d "$fallback_bst" ]; then
    bst_search_path="$bst_search_path:$fallback_bst"
fi
export BSTINPUTS="$bst_search_path:${BSTINPUTS:-}"

missing_packages=""
for package_file in \
    placeins.sty siunitx.sty wallpaper.sty nomencl.sty lastpage.sty \
    enumitem.sty mdframed.sty comment.sty zref-abspage.sty needspace.sty
do
    if ! kpsewhich "$package_file" >/dev/null 2>&1; then
        missing_packages="$missing_packages $package_file"
    fi
done

if [ -n "$missing_packages" ]; then
    printf '%s\n' "Paquets LaTeX introuvables :$missing_packages" >&2
    printf '%s\n' \
        "Installe une distribution TeX Live complete ou les paquets correspondants, puis relance ./compile.sh." >&2
    exit 1
fi

mkdir -p "$build_dir" "$output_dir"

# Une interruption pendant l'écriture du .aux peut le laisser tronqué et
# empêcher toute compilation ultérieure. Ces artefacts sont entièrement
# régénérés par la séquence pdflatex/bibtex ci-dessous ; les sources et PDF de
# livraison ne sont donc pas concernés par ce nettoyage ciblé.
rm -f "$build_dir/rapportCS.aux" "$build_dir/rapportCS.bbl" \
    "$build_dir/rapportCS.blg" "$build_dir/rapportCS.log" \
    "$build_dir/rapportCS.out" "$build_dir/rapportCS.toc" \
    "$build_dir/rapportCS.nlo"
# Une ancienne compilation lancée sans -output-directory peut aussi laisser
# ces mêmes fichiers à la racine ; TeX les relirait via TEXINPUTS.
rm -f "$project_dir/rapportCS.aux" "$project_dir/rapportCS.bbl" \
    "$project_dir/rapportCS.blg" "$project_dir/rapportCS.log" \
    "$project_dir/rapportCS.out" "$project_dir/rapportCS.toc" \
    "$project_dir/rapportCS.nlo"

cd "$project_dir"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$build_dir" rapportCS.tex

cd "$build_dir"
bibtex rapportCS

cd "$project_dir"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$build_dir" rapportCS.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$build_dir" rapportCS.tex

cp "$build_dir/rapportCS.pdf" "$output_dir/rapportCS.pdf"
cp "$build_dir/rapportCS.pdf" "$project_dir/rapportCS.pdf"
printf '%s\n' "PDF cree : $output_dir/rapportCS.pdf"
printf '%s\n' "Copie pour l'editeur : $project_dir/rapportCS.pdf"
