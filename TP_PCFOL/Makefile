all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_pcfol" "\input{infoh410_tp_pcfol.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_pcfol" "\input{infoh410_tp_pcfol.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_pcfol_correction" "\def\correction{} \input{infoh410_tp_pcfol.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_pcfol_correction" "\def\correction{} \input{infoh410_tp_pcfol.tex}"

clean:
	rm -f infoh410_tp_pcfol_correction.pdf infoh410_tp_pcfol.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
