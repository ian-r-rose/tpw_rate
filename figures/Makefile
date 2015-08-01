all: eigengap.pdf misfit.pdf perturb.pdf

eigengap.pdf: eigengap.py
	python eigengap.py

misfit.pdf: misfit.py
	python misfit.py

perturb.pdf: perturb.sh
	./perturb.sh

.PHONY: clean
clean:
	rm -f *.pdf *.png
