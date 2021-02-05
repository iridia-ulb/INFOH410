all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp1" "\input{infoh410_tp1.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp1" "\input{infoh410_tp1.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp1_correction" "\def\correction{} \input{infoh410_tp1.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp1_correction" "\def\correction{} \input{infoh410_tp1.tex}"

clean:
	rm -f infoh410_tp1_correction.pdf infoh410_tp1.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
