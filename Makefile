texargs = -interaction nonstopmode -halt-on-error -file-line-error

.PHONY: figures
figures:
	make -C figures

.PHONY: clean
clean:
	rm *.bbl *.blg *.aux *.log tpw_rate.pdf

