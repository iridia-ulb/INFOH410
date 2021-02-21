all: labo correction
	echo "done"


labo: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_dtb" "\input{infoh410_tp_dtb.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_dtb" "\input{infoh410_tp_dtb.tex}"


correction: *.tex
	pdflatex -shell-escape -jobname="infoh410_tp_dtb_correction" "\def\correction{} \input{infoh410_tp_dtb.tex}"
	pdflatex -shell-escape -jobname="infoh410_tp_dtb_correction" "\def\correction{} \input{infoh410_tp_dtb.tex}"

clean:
	rm -f infoh410_tp_dtb_correction.pdf infoh410_tp_dtb.pdf
	rm -f *.idx *.aux *.log *.nav *.out *.snm *.toc
