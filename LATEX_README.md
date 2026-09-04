# TRACE LaTeX manuscript

`cas-sc-sample.tex` is the main manuscript source. Despite the historical file
name, it uses Elsevier's `cas-dc` class and therefore produces the double-column
layout used for the current Information Fusion submission draft.

The repository includes the required class, bibliography style, bibliography,
method figure, and CAS thumbnail assets. A current compiled copy is available as
`cas-sc-sample.pdf`.

Compile with Tectonic:

```sh
tectonic cas-sc-sample.tex
```

Alternatively, use the standard XeLaTeX/BibTeX sequence:

```sh
xelatex cas-sc-sample.tex
bibtex cas-sc-sample
xelatex cas-sc-sample.tex
xelatex cas-sc-sample.tex
```

Temporary LaTeX products are ignored by Git. The submitted source should retain
`cas-sc-sample.tex`, `cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst`,
`cas-refs.bib`, `Figures/TRACE.pdf`, and the `thumbnails/` directory.
