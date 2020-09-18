# CSAF Software Requirements Specification (SRS)

A SRS document was written for CSAF to plan the necessary features for known user workflows. The current implementation of CSAF has many of the features enumerated in the document, but is incomplete. **Do not use this document as a user manual.**

## Installation

To build a new copy of the SRS, a standard `pdflatex` suffices. In order to get all of the passes done at once, `latexmk` is recommended,

```
latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make csaf.tex
```

The PDF `csaf.pdf` should be available for perusal after the build.