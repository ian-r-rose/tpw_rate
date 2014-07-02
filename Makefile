tpw_rate.pdf: tpw_rate.tex tpw_rate.bib
	pdflatex tpw_rate
	bibtex tpw_rate
	pdflatex tpw_rate
	pdflatex tpw_rate

clean:
	rm *.bbl *.blg *.aux *.log tpw_rate.pdf

figures: figures/*
	cd figures && ./perturb.sh
	cd figures && python misfit.py
	cd figures && python eigengap.py
