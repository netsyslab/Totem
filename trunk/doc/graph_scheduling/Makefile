
SOURCE := $(wildcard *model.tex)
SUFFIX := $(patsubst %.tex,%,$(SOURCE))

all: pdf bib pdf2 pdf3 

pdf: 
	pdflatex $(SOURCE)

bib: 
	bibtex $(SUFFIX)

pdf2:
	pdflatex $(SOURCE)

pdf3:
	pdflatex $(SOURCE)

html:
	latex2html $(SOURCE)
	mv $(SUFFIX) html

clean: 
	rm -vf *.aux *.blg *.bbl *.log *.dvi *.toc *~ *.pdf
	rm -rf html/

