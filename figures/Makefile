all: eigengap.pdf misfit.pdf perturb.pdf reference_frames.pdf

eigengap.pdf: eigengap.py
	python eigengap.py

misfit.pdf: misfit.py
	python misfit.py

perturb.pdf: perturb.py
	python perturb.py

reference_frames.pdf: reference_frames.py
	python reference_frames.py

.PHONY: clean
clean:
	rm -f *.pdf *.png
