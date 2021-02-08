all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp4" "\input{infoh410_tp4.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp4" "\input{infoh410_tp4.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp4_correction" "\def\correction{} \input{infoh410_tp4.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp4_correction" "\def\correction{} \input{infoh410_tp4.tex}"

clean:
	rm -f infoh410_tp4_correction.pdf infoh410_tp4.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
