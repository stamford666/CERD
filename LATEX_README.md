# CERD manuscript package

The archival manuscript bundle is `CERD_LaTeX_current.zip`. Its entry point is
`cas-sc-sample.tex`; all class/style, bibliography, thumbnail, and figure files
needed by the source are included in the archive.

The experiment tables use arithmetic mean and sample standard deviation across
three independently trained seeds (ADNI: 0/1/2; ABCD: 31/32/33). No row is a
probability ensemble. The current ABCD experiment is the family-disjoint ADHD-
presentation endpoint with direct QC/LD-pruned SNP dosages and approximately
15% incomplete participants.

Compile with a standard TeX distribution using:

```bash
pdflatex cas-sc-sample.tex
bibtex cas-sc-sample
pdflatex cas-sc-sample.tex
pdflatex cas-sc-sample.tex
```

Tectonic can also compile the package directly:

```bash
tectonic cas-sc-sample.tex
```
