$pdf_mode = 1;
$out_dir = 'output';
$aux_dir = 'output';

$ENV{'BSTINPUTS'} = 'bst:' . ($ENV{'BSTINPUTS'} // '');

$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex = 'bibtex %O %B';

$clean_ext = 'synctex.gz run.xml';
