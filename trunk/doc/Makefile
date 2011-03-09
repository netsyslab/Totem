
SOURCE := $(wildcard *571_midterm.tex)
SUFFIX := $(patsubst %.tex,%,$(SOURCE))

all: pdf

pdf: 
	pdflatex $(SOURCE)

bib: 
	bibtex $(SUFFIX)

html:
	latex2html $(SOURCE)
	mv $(SUFFIX) html

clean:
	@rm -f *.aux *.blg *.bbl *.log *.dvi *.toc *~
	@rm -rf html/

clean-all:
	@rm -f *.aux *.blg *.bbl *.log *.dvi *.toc *.pdf *~
	@rm -rf html/
