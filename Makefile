tpw_rate.pdf: tpw_rate.tex tpw_rate.bib figures
	pdflatex tpw_rate
	bibtex tpw_rate
	pdflatex tpw_rate
	pdflatex tpw_rate

.PHONY: figures
figures:
	make -C figures

.PHONY: clean
clean:
	rm *.bbl *.blg *.aux *.log tpw_rate.pdf

