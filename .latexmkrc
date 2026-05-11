$pdf_mode = 1;
$out_dir = 'docs/paper/output';
$aux_dir = 'docs/paper/output';
$bibtex_use = 2;
@default_files = ('docs/paper/sn-article.tex');

use Cwd 'getcwd';
my $repo_root = getcwd();
$ENV{'TEXINPUTS'} = "$repo_root/docs/paper//:" . ($ENV{'TEXINPUTS'} // '');
$ENV{'BIBINPUTS'} = "$repo_root/docs/paper//:" . ($ENV{'BIBINPUTS'} // '');
$ENV{'BSTINPUTS'} = "$repo_root/docs/paper//:" . ($ENV{'BSTINPUTS'} // '');

$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';
$bibtex = 'bibtex %O %B';
