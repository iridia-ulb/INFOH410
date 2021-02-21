all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_s" "\input{infoh410_tp_s.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_s" "\input{infoh410_tp_s.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_s_correction" "\def\correction{} \input{infoh410_tp_s.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_s_correction" "\def\correction{} \input{infoh410_tp_s.tex}"

clean:
	rm -f infoh410_tp_s_correction.pdf infoh410_tp_s.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
