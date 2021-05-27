all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_gaa" "\input{infoh410_tp_gaa.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_gaa" "\input{infoh410_tp_gaa.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_gaa_correction" "\def\correction{} \input{infoh410_tp_gaa.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_gaa_correction" "\def\correction{} \input{infoh410_tp_gaa.tex}"

clean:
	rm -f infoh410_tp_gaa_correction.pdf infoh410_tp_gaa.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
