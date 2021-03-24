all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_nn" "\input{infoh410_tp_nn.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_nn" "\input{infoh410_tp_nn.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_nn_correction" "\def\correction{} \input{infoh410_tp_nn.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_nn_correction" "\def\correction{} \input{infoh410_tp_nn.tex}"

clean:
	rm -f infoh410_tp_nn_correction.pdf infoh410_tp_nn.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
