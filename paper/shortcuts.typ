// General
#let limn = $lim_(n->infinity)$
#let int(a, b) = $integral_(#a)^(#b)$
#let lint(a) = $integral_(#a)$
#let ub(a, b) = $underbrace(#a, #b)$
#let inf = $infinity$
#let ind = $bb(1)$
#let pm = $plus.minus$
#let dv(a) = $(dif)/(dif #a)$
#let pdv(a) = $(partial)/(partial #a)$
#let ddv(a, b) = $(dif #a)/(dif #b)$
#let pddv(a, b) = $(partial #a)/(partial #b)$
#let drd(i, j) = $delta_(#i #j)$
#let tp = $times.o$

// Sums
#let sumo(a, b) = $sum_(#a=1)^#b$
#let suma(a, i, b) = $sum_(#a=#i)^#b$
#let sumin = $sum_(i=1)^n$
#let sumjn = $sum_(j=1)^n$
#let sumkn = $sum_(k=1)^n$
#let sumi = $sum_(i=1)^infinity$
#let sumj = $sum_(i=1)^infinity$
#let sumk = $sum_(k=1)^infinity$

// Probability & Statistics
#let chis(n) = $chi^2_((#n))$
#let avg(a) = $macron(#a)$
#let mgf(a, t) = $M_(#a) (#t)$

// Analysis
#let eps = $epsilon$
#let borel = $frak(B)_RR$
#let infim = $"inf"$
#let suprem = $"sup"$
#let RRR = $overline(RR)$
#let im = $"Im"$


// Formatting
#let lrs(x) = $lr(#x, size: #200%)$
#let bf(x) = $upright(bold(#x))$

// Fix some built-in symbols
#let exists = $thin exists thin$
#let forall = $thin forall thin$
#let nothing = $diameter$
