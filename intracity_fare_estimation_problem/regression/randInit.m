function W = randInitializeWeights(L_in, L_out)
 
ep=0.12;
W = rand(L_out, L_in)*(2*ep)-ep;

end
