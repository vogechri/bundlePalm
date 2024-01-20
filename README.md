# bundlePalm

partitioning cameras, landmarks go to BOTH parts (middle of edge = lm) edges are 2 residuals.

1) 1/2 r_i for DRS?
2) best fill up as what i had before for PALM
3) double check: f(x1,..,xi,..xn) -> wrt xi.
4) drs sum_fi()
is different significantly
fi(x1,x2) -> deriv wrt to both.
1/2 fi(x1,x2) -> still to both .. 
1/2(fi(const, x2) + fi(x1,const)) 
gain to split landmark could be to get FULL update locally per lm, problem need to add cams as well