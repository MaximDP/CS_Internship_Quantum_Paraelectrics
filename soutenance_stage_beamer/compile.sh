#!/bin/sh

# Compilation portable de la soutenance sans dépendre de latexmk.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir="$project_dir/build"

mkdir -p "$build_dir"

# Ces fichiers sont intégralement régénérés. Les supprimer évite qu'un .aux
# tronqué après une interruption bloque les compilations suivantes.
rm -f "$build_dir/main.aux" "$build_dir/main.bbl" "$build_dir/main.blg" \
    "$build_dir/main.log" "$build_dir/main.out" "$build_dir/main.toc" \
    "$build_dir/main.nav" "$build_dir/main.snm"

cd "$project_dir"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$build_dir" main.tex

cd "$project_dir"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$build_dir" main.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$build_dir" main.tex

cp "$build_dir/main.pdf" "$project_dir/soutenance_stage.pdf"
printf '%s\n' "PDF créé : $project_dir/soutenance_stage.pdf"
