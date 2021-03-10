all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_lsmm" "\input{infoh410_tp_lsmm.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_lsmm" "\input{infoh410_tp_lsmm.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_lsmm_correction" "\def\correction{} \input{infoh410_tp_lsmm.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_lsmm_correction" "\def\correction{} \input{infoh410_tp_lsmm.tex}"

clean:
	rm -f infoh410_tp_lsmm_correction.pdf infoh410_tp_lsmm.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
