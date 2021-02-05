all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp3" "\input{infoh410_tp3.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp3" "\input{infoh410_tp3.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp3_correction" "\def\correction{} \input{infoh410_tp3.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp3_correction" "\def\correction{} \input{infoh410_tp3.tex}"

clean:
	rm -f infoh410_tp3_correction.pdf infoh410_tp3.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
