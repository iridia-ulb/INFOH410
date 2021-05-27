all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_rl" "\input{infoh410_tp_rl.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_rl" "\input{infoh410_tp_rl.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_rl_correction" "\def\correction{} \input{infoh410_tp_rl.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_rl_correction" "\def\correction{} \input{infoh410_tp_rl.tex}"

clean:
	rm -f infoh410_tp_rl_correction.pdf infoh410_tp_rl.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
